"""Evaluate online PoVo segmentation using previously generated image vocabularies.

Reuse RGB-only VLM responses without loading another large language model.
Ground truth is used only for evaluation. Semantic labels use exact
ScanNet200 names; aliases and out-of-taxonomy names remain unknown.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from datasets.data_loader import ScanReader
from exp.eval_inst import BaseEval
from exp.scannet200_splits import INSTANCE_CAT_SCANNET_200 as NAMES
from libs.inference import select_camera_frames
from models.povo import SAM2Mask
from models.vocabulary import parse_object_names
from tools.discover_vocabulary import save_json
from tools.recognize_instances import evaluation_metrics, detailed_metrics, save_tensor


def namespace(value):
    return SimpleNamespace(**{key: namespace(item) for key, item in value.items()}) if isinstance(value, dict) else value


def image_hash(image):
    return hashlib.sha256(image.convert('RGB').resize((640, 480)).tobytes()).hexdigest()


def taxonomy_class_ids(names, classes):
    lookup = {name: index for index, name in enumerate(NAMES)}
    mapping = [lookup.get(name, -1) for name in names]
    return torch.tensor([mapping[index] if 0 <= index < len(mapping) else -1
                         for index in torch.as_tensor(classes).tolist()], dtype=torch.long)


def cached_vocabulary(root, data_root, scenes, reader):
    labels_by_image, selected, metadata, digests = {}, {}, None, {}
    for scene in scenes:
        path = root / f'{scene}.json'
        record = json.loads(path.read_text())
        if metadata is None:
            metadata = record['metadata']
        elif metadata != record['metadata']:
            raise ValueError('cached vocabularies have different model or generation settings')
        expected = select_camera_frames(reader.get_image_dirs(scene, 'val'), reader, metadata['frames'])
        expected = [str(Path(path).relative_to(data_root)) for path in expected]
        selected[scene] = expected
        if set(expected) != {item['image'] for item in record['images']}:
            raise ValueError(f'{scene}: cached vocabulary image selection differs from the dataset')
        for item in record['images']:
            if parse_object_names(item['response']) != item['labels']:
                raise ValueError(f'{scene}: saved labels differ from the current parser')
            digest = image_hash(reader.get_image(str(data_root / item['image'])))
            if digest != item.get('rgb_sha256'):
                raise ValueError(f'{scene}: image content does not match the saved vocabulary')
            if digest in labels_by_image and labels_by_image[digest] != item['labels']:
                raise ValueError('identical RGB images have inconsistent vocabularies')
            labels_by_image[digest] = item['labels']
        digests[scene] = hashlib.sha256(path.read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
    if fingerprint != metadata['selection_sha256']:
        raise ValueError('scene list differs from the vocabulary-generation cohort')
    return labels_by_image, metadata, digests


def report(args, scenes, reader, metadata):
    evaluator = BaseEval(NAMES, use_label=True, dataset_name='scannet200')
    localization = BaseEval(NAMES, use_label=False, dataset_name='scannet200')
    semantic_matches, localization_matches, per_scene = {}, {}, {}
    for scene in scenes:
        saved = torch.load(args.output_root / f'{scene}.pth', map_location='cpu', weights_only=False)
        if saved['metadata'] != metadata:
            raise ValueError(f'{scene}: predictions have different inference settings')
        assignment = saved['assignment'].numpy()
        _, _, instances_gt, semantic_gt = reader.get_pcd(scene, 'val')
        predictions, local_predictions = [], []
        for index, label in enumerate(saved['class_ids']):
            mask = (assignment == index + 1).astype(np.uint8)
            if label >= 0 and saved['scores'][index] > 0:
                predictions.append({'scan_id': scene, 'label_id': int(label) + 1,
                                    'conf': float(saved['scores'][index]), 'pred_mask': mask})
            if saved['objectness'][index] > 0:
                local_predictions.append({'scan_id': scene, 'label_id': 1,
                                          'conf': float(saved['objectness'][index]), 'pred_mask': mask})
        gt, pred = evaluator.assign_instances_for_scan(predictions, semantic_gt.numpy(), instances_gt.numpy())
        semantic_matches[scene] = {'gt': gt, 'pred': pred}
        gt, pred = localization.assign_instances_for_scan(local_predictions, semantic_gt.numpy(), instances_gt.numpy())
        localization_matches[scene] = {'gt': gt, 'pred': pred}
        per_scene[scene] = {'instances': len(saved['class_ids']),
                            'known_foreground_instances': int((saved['class_ids'] >= 0).sum()),
                            'seconds_without_vlm': saved['seconds_without_vlm'],
                            'raw_assignment_sha256': saved['raw_assignment_sha256']}
    ap, recall = evaluator.evaluate_matches(semantic_matches)
    local_ap, local_recall = localization.evaluate_matches(localization_matches)
    summary = {'metadata': metadata, 'scenes': scenes, 'frames': metadata['vocabulary']['frames'],
               'scope': 'Online vocabulary -> Grounding DINO -> SAM -> ViT-B/32 point fusion -> retained clustering',
               'mapping': 'Exact ScanNet200 names; aliases and out-of-taxonomy names remain unknown',
               'semantic': evaluation_metrics(evaluator, ap, recall),
               'class_agnostic': evaluation_metrics(localization, local_ap, local_recall),
               'semantic_details': detailed_metrics(evaluator, ap, recall),
               'per_scene': per_scene,
               'instances': sum(item['instances'] for item in per_scene.values()),
               'known_foreground_instances': sum(item['known_foreground_instances'] for item in per_scene.values()),
               'seconds_without_vlm': sum(item['seconds_without_vlm'] for item in per_scene.values())}
    save_json(args.output_root / 'summary.json', summary)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ['metadata', 'per_scene', 'semantic_details']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--vocabulary-root', type=Path, required=True)
    parser.add_argument('--scenes-file', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/scannet200_llava.yaml')
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    scenes = args.scenes_file.read_text().splitlines()
    if not scenes or len(scenes) != len(set(scenes)):
        parser.error('scene IDs must be nonempty and unique')
    reader = ScanReader(str(args.data_root), 'cpu')
    labels, vocabulary_metadata, vocabulary_digests = cached_vocabulary(
        args.vocabulary_root, args.data_root, scenes, reader)
    config_dict = yaml.safe_load(args.config.read_text())
    repository = Path(__file__).resolve().parents[1]
    source_paths = ([Path(__file__), repository / 'models/povo.py', repository / 'models/vocabulary.py',
                     repository / 'datasets/data_loader.py'] + sorted((repository / 'libs').glob('*.py')))
    data_paths = []
    for scene in scenes:
        data_paths.extend([args.data_root / 'scannet200/val' / f'{scene}.pth',
                           args.data_root / 'intrinsic' / f'{scene}.txt'])
        saved = json.loads((args.vocabulary_root / f'{scene}.json').read_text())
        for item in saved['images']:
            parent = (args.data_root / item['image']).parent.parent
            frame = Path(item['image']).stem
            data_paths.extend([parent / 'depth' / f'{frame}.png', parent / 'pose' / f'{frame}.txt'])
    def digest_file(path):
        hasher = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    metadata = {'config': config_dict, 'vocabulary': vocabulary_metadata,
                'vocabulary_sha256': vocabulary_digests,
                'dataset_sha256': {str(path.relative_to(args.data_root)): digest_file(path) for path in data_paths},
                'source_sha256': {str(path.relative_to(repository)): digest_file(path) for path in source_paths}}
    args.output_root.mkdir(parents=True, exist_ok=True)
    pending = []
    for scene in scenes:
        path = args.output_root / f'{scene}.pth'
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=False)
            if saved['metadata'] != metadata:
                raise ValueError(f'{scene}: saved predictions have different inference settings')
        else:
            pending.append(scene)
    if pending:
        torch.manual_seed(0)
        model = SAM2Mask(namespace(config_dict), device='cuda',
                         data_reader=ScanReader(str(args.data_root), 'cuda'),
                         image_vocabulary=lambda image: labels[image_hash(image)])
        for scene in pending:
            torch.cuda.synchronize()
            started = time.perf_counter()
            masks, scores, _, _, _ = model(scene, k_img=vocabulary_metadata['frames'])
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            raw_assignment = ((masks.int().argmax(0) + 1) * masks.bool().any(0)).cpu().int() if len(masks) else torch.zeros(masks.shape[1], dtype=torch.int32)
            raw_hash = hashlib.sha256(raw_assignment.numpy().tobytes()).hexdigest()
            keep = masks.sum(1) >= 100
            classes = taxonomy_class_ids(model.discovered_class_names, model.instance_class_ids)[keep.cpu()]
            masks = masks[keep].bool()
            if len(masks):
                if (masks.sum(0) > 1).any():
                    raise ValueError('instance masks must be disjoint for assignment export')
                assignment = ((masks.int().argmax(0) + 1) * masks.any(0)).cpu().int()
            else:
                assignment = torch.zeros(masks.shape[1], dtype=torch.int32)
            save_tensor(args.output_root / f'{scene}.pth',
                        {'metadata': metadata, 'assignment': assignment, 'class_ids': classes,
                         'scores': scores[keep].float().cpu(),
                         'objectness': model.instance_objectness[keep].float().cpu(),
                         'class_names': list(NAMES), 'discovered_vocabulary': model.discovered_class_names,
                         'raw_assignment_sha256': raw_hash, 'seconds_without_vlm': elapsed})
            print(f'{scene}: {len(masks)} instances, {int((classes >= 0).sum())} known labels, {elapsed:.2f}s', flush=True)
    report(args, scenes, reader, metadata)


if __name__ == '__main__':
    main()
