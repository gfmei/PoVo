"""Discover free image vocabularies and evaluate visible ScanNet class recall.

The model receives RGB images only. Annotation projection happens in report,
after all responses have been saved, and never supplies generation candidates.
This measures vocabulary discovery, not instance segmentation AP.
"""

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from datasets.data_loader import ScanReader
from datasets.meta_data.scannet200_constants import CLASS_LABELS_200
from libs.inference import parse_frame_count, resize_depth_and_intrinsic, select_camera_frames
from libs.mapper import PointCloudToImageMapper
from models.vocabulary import (DEFAULT_MODEL, DEFAULT_REVISION, OBJECT_PROMPT,
                               REPETITION_PENALTY, NO_REPEAT_NGRAM_SIZE,
                               MAX_CATEGORIES, PARSER_VERSION,
                               discover_image_categories, load_vocabulary_model, parse_object_names)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def vocabulary_recall(predicted, visible):
    """Exact category agreement; preserve distinctions such as desk/table."""
    predicted, visible = set(predicted), set(visible)
    return {'matched': len(predicted & visible), 'eligible': len(visible),
            'recall': len(predicted & visible) / len(visible) if visible else None}


def validate_cached_image(record, image):
    if record.get('rgb_sha256') != hashlib.sha256(image.convert('RGB').resize((640, 480)).tobytes()).hexdigest():
        raise ValueError('RGB image differs from the cached vocabulary; regenerate discovery.')
    if parse_object_names(record['response']) != record['labels']:
        raise ValueError('cached category labels differ from the current parser')


def discover(args, records, metadata, reader):
    pending = []
    for scene, images in records.items():
        path = args.output_root / f'{scene}.json'
        existing = json.loads(path.read_text()) if path.exists() else None
        if existing is not None and existing['metadata'] != metadata:
            raise ValueError(f'{path}: model or generation settings differ; use a separate output directory.')
        completed = {} if existing is None else {item['image']: item for item in existing['images']}
        if any(item not in {str(image.relative_to(args.data_root)) for image in images} for item in completed):
            raise ValueError(f'{path}: saved image selection differs.')
        for image in images:
            relative = str(image.relative_to(args.data_root))
            if relative in completed:
                validate_cached_image(completed[relative], reader.get_image(str(image)))
        pending.extend((scene, image, completed) for image in images
                       if str(image.relative_to(args.data_root)) not in completed)
    if not pending:
        return
    model, processor = load_vocabulary_model(
        args.model_id, args.device, revision=args.revision, cache_dir=args.cache_dir,
        load_in_4bit=not args.full_precision, local_files_only=args.local_files_only)
    for index, (scene, image_path, completed) in enumerate(pending, 1):
        torch.cuda.synchronize() if torch.device(args.device).type == 'cuda' else None
        started = time.perf_counter()
        image = reader.get_image(str(image_path)).resize((640, 480))
        labels, responses = discover_image_categories(
            [image], model, processor,
            max_new_tokens=args.max_new_tokens, return_responses=True)
        torch.cuda.synchronize() if torch.device(args.device).type == 'cuda' else None
        relative = str(image_path.relative_to(args.data_root))
        completed[relative] = {'image': relative, 'labels': labels[0], 'response': responses[0],
                               'rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                               'seconds': time.perf_counter() - started}
        save_json(args.output_root / f'{scene}.json', {'metadata': metadata,
                                                      'images': list(completed.values())})
        print(f'{index}/{len(pending)} {scene}/{image_path.stem}: {labels[0]} '
              f'({completed[relative]["seconds"]:.2f}s)', flush=True)


def report(args, records, metadata, reader):
    mapper = PointCloudToImageMapper([640, 480], cut_bound=10)
    scene_counts, image_counts, timings, label_counts = [], [], [], []
    foreground_counts = []
    known_counts, foreground_known_counts = [], []
    categories = set(CLASS_LABELS_200)
    per_scene, failed = {}, []
    for scene, expected_images in records.items():
        saved = json.loads((args.output_root / f'{scene}.json').read_text())
        if saved['metadata'] != metadata:
            raise ValueError(f'{scene}: report settings differ from saved discovery.')
        lookup = {item['image']: item for item in saved['images']}
        if set(lookup) != {str(image.relative_to(args.data_root)) for image in expected_images}:
            raise ValueError(f'{scene}: incomplete or different image selection.')
        points, _, _, labels = reader.get_pcd(scene, 'val')
        visible_scene, predicted_scene = set(), set()
        for image_path in expected_images:
            item = lookup[str(image_path.relative_to(args.data_root))]
            validate_cached_image(item, reader.get_image(str(image_path)))
            depth, intrinsic = resize_depth_and_intrinsic(
                reader.get_depth(str(image_path)).astype('float32') / 1000.,
                reader.get_intrinsic(scene), [640, 480])
            mapping = mapper.compute_mapping_torch(
                reader.get_pose(str(image_path)), points, depth, intrinsic=intrinsic)
            visible_labels, counts = labels[mapping[:, 2].bool()].unique(return_counts=True)
            visible = {CLASS_LABELS_200[int(label)] for label, count in zip(visible_labels, counts)
                       if 0 <= int(label) < len(CLASS_LABELS_200) and int(count) >= args.min_visible_points}
            image_counts.append(vocabulary_recall(item['labels'], visible))
            foreground_counts.append(vocabulary_recall(item['labels'], visible - {'wall', 'floor'}))
            known = set(item['labels']) & categories
            known_counts.append(len(known))
            foreground_known_counts.append(len(known - {'wall', 'floor'}))
            visible_scene.update(visible)
            predicted_scene.update(item['labels'])
            timings.append(item['seconds'])
            label_counts.append(len(item['labels']))
            if not item['labels']:
                failed.append(item['image'])
        counts = vocabulary_recall(predicted_scene, visible_scene)
        scene_counts.append(counts)
        per_scene[scene] = dict(counts, predicted=sorted(predicted_scene), visible=sorted(visible_scene),
                               missing=sorted(visible_scene - predicted_scene))
    def aggregate(values):
        matched = sum(item['matched'] for item in values)
        eligible = sum(item['eligible'] for item in values)
        return dict(matched=matched, eligible=eligible, recall=matched / eligible if eligible else None)
    def precision(values, predicted):
        matched, count = sum(item['matched'] for item in values), sum(predicted)
        return dict(matched=matched, predicted=count, precision=matched / count if count else None)
    summary = {'metadata': metadata, 'scope': 'Image vocabulary discovery; not segmentation AP',
               'matching': 'Exact ScanNet200 category names; no synonym merging or candidate vocabulary',
               'visibility': f'At least {args.min_visible_points} depth-visible projected points per class per image',
               'scenes': len(records), 'images': len(timings),
               'image_class_recall': aggregate(image_counts),
               'foreground_image_class_recall': aggregate(foreground_counts),
               'image_taxonomy_precision': precision(image_counts, known_counts),
               'foreground_image_taxonomy_precision': precision(foreground_counts, foreground_known_counts),
               'precision_scope': 'Only predictions matching a ScanNet200 category name; unrecognized names are excluded',
               'known_category_fraction': sum(known_counts) / sum(label_counts) if sum(label_counts) else None,
               'scene_union_class_recall': aggregate(scene_counts),
               'nonempty_response_rate': 1 - len(failed) / len(timings),
               'mean_labels_per_image': sum(label_counts) / len(label_counts),
               'mean_seconds_per_image': sum(timings) / len(timings),
               'generation_seconds': sum(timings), 'empty_responses': failed, 'per_scene': per_scene}
    save_json(args.output_root / 'summary.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key != 'per_scene'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--scenes-file', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--frames', type=parse_frame_count, default=50,
                        help='views per scene (default: 50); all uses every finite-pose prepared view')
    parser.add_argument('--model-id', default=DEFAULT_MODEL)
    parser.add_argument('--revision')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--full-precision', action='store_true')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--min-visible-points', type=int, default=100)
    parser.add_argument('--phase', choices=['discover', 'report', 'all'], default='all')
    args = parser.parse_args()
    if args.min_visible_points < 1:
        parser.error('min-visible-points must be positive')
    args.data_root = args.data_root.resolve()
    if args.model_id == DEFAULT_MODEL and args.revision is None:
        args.revision = DEFAULT_REVISION
    if args.revision is None:
        parser.error('pin --revision for model comparisons')
    scenes = args.scenes_file.read_text().splitlines()
    if not scenes or len(scenes) != len(set(scenes)):
        parser.error('scenes-file must contain unique scene IDs')
    reader = ScanReader(str(args.data_root), 'cpu')
    records = {scene: [Path(image) for image in select_camera_frames(
        reader.get_image_dirs(scene, 'val'), reader, args.frames)] for scene in scenes}
    if any(not images or (args.frames > 0 and len(images) != args.frames) for images in records.values()):
        parser.error('every scene must have enough images with valid camera poses')
    selected = {scene: [str(image.relative_to(args.data_root)) for image in images]
                for scene, images in records.items()}
    try:
        fla_version = importlib.metadata.version('flash-linear-attention')
    except importlib.metadata.PackageNotFoundError:
        fla_version = None
    metadata = {'model_id': args.model_id, 'revision': args.revision,
                'precision': 'native' if args.full_precision else 'NF4 double quantization',
                'prompt': OBJECT_PROMPT, 'enable_thinking': False, 'do_sample': False,
                'repetition_penalty': REPETITION_PENALTY, 'no_repeat_ngram_size': NO_REPEAT_NGRAM_SIZE,
                'max_categories': MAX_CATEGORIES, 'parser_version': PARSER_VERSION,
                'max_new_tokens': args.max_new_tokens, 'frames': args.frames,
                'image_size': [640, 480],
                'torch': torch.__version__,
                'transformers': importlib.metadata.version('transformers'),
                'flash_linear_attention': fla_version,
                'selection_sha256': hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()}
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.phase in ['discover', 'all']:
        discover(args, records, metadata, reader)
    if args.phase in ['report', 'all']:
        report(args, records, metadata, reader)


if __name__ == '__main__':
    main()
