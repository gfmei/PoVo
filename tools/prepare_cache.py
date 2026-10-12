"""Prepare fixed ScanNet caches for the retained PoVo inference workflow."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import clip
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from datasets.data_loader import ScanReader
from datasets.preprocess.SensorData import SensorData
from datasets.scannet.preprocess_scannet200 import handle_process
from libs.inference import parse_frame_count
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2


def diverse_frames(poses, valid, count):
    """Greedily cover camera position and viewing direction; ignore all labels."""
    if count == 0:
        return [int(frame) for frame in valid]
    selected = [int(valid[0])]
    position = poses[valid, :3, 3]
    forward = poses[valid, :3, 2]
    forward /= np.maximum(np.linalg.norm(forward, axis=1, keepdims=True), 1e-9)
    features = np.concatenate((position / 0.5, forward * 2), axis=1)
    distance = np.full(len(valid), np.inf)
    reverse = {int(frame): i for i, frame in enumerate(valid)}
    for frame in selected:
        distance = np.minimum(distance, ((features - features[reverse[frame]]) ** 2).sum(1))
    while len(selected) < min(count, len(valid)):
        distance[[reverse[frame] for frame in selected]] = -1
        index = int(distance.argmax())
        selected.append(int(valid[index]))
        distance = np.minimum(distance, ((features - features[index]) ** 2).sum(1))
    return selected


def full_validation_scenes(raw_root, split_path):
    """Require every listed scan, including repeated scans of the same room."""
    scenes = split_path.read_text().splitlines()
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError('validation split must contain unique scene IDs')
    suffixes = ['.sens', '_vh_clean_2.ply', '_vh_clean_2.0.010000.segs.json', '.aggregation.json']
    missing = [str(raw_root / 'scans' / scene / (scene + suffix))
               for scene in scenes for suffix in suffixes
               if not (raw_root / 'scans' / scene / (scene + suffix)).is_file()]
    if missing:
        raise FileNotFoundError('full validation inputs are missing: ' + ', '.join(missing))
    return scenes


def scene_cache_complete(root, scene, selected):
    paths = [root / 'scannet200/val' / f'{scene}.pth', root / 'intrinsic' / f'{scene}.txt']
    for index in selected:
        paths.extend([root / 'scannet_2d/val' / scene / folder / f'{index}.{suffix}'
                      for folder, suffix in [('color', 'jpg'), ('depth', 'png'), ('pose', 'txt')]])
        paths.append(root / 'masks/val' / scene / 'sam2' / f'{index}.pth')
    return bool(selected) and all(path.is_file() for path in paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--held-out-scenes', type=int, default=20)
    parser.add_argument('--full-validation', action='store_true', help='prepare every official validation scan')
    parser.add_argument('--frames', type=parse_frame_count, default=50,
                        help='views per scene (default: 50); all uses every finite-pose view')
    parser.add_argument('--seed', type=int, default=20261010)
    args = parser.parse_args()
    if (0 < args.frames < 8) or args.held_out_scenes < 1:
        parser.error('at least eight frames (or all) and one held-out scene are required')
    repository = Path(__file__).resolve().parents[1]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / 'manifest.json'
    development = ['scene0304_00', 'scene0011_00']
    validation = full_validation_scenes(args.raw_root, repository / 'datasets/meta_data/scannetv2_val.txt') if args.full_validation else None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['seed'] != args.seed or manifest['target_frames'] != args.frames:
            parser.error('existing cache has a different seed or frame count')
        if args.full_validation != ('validation_scenes' in manifest) or (args.full_validation and manifest['validation_scenes'] != validation):
            parser.error('existing cache uses a different scene selection; choose a fresh output root')
    else:
        split = (repository / 'datasets/meta_data/scannetv2_val.txt').read_text().splitlines()
        available = []
        for scene in sorted(split):
            root = args.raw_root / 'scans' / scene
            suffixes = ['.sens', '_vh_clean_2.ply', '_vh_clean_2.0.010000.segs.json', '.aggregation.json']
            if scene[:9] not in {name[:9] for name in development} and all((root / (scene + suffix)).is_file() for suffix in suffixes):
                available.append(scene)
        rng = np.random.default_rng(args.seed)
        held_out, rooms = [], set()
        for scene in rng.permutation(available):
            room = scene[:9]
            if room not in rooms:
                held_out.append(str(scene))
                rooms.add(room)
            if len(held_out) == args.held_out_scenes:
                break
        if not args.full_validation and len(held_out) != args.held_out_scenes:
            parser.error('not enough independent validation scenes')
        manifest = {'seed': args.seed, 'target_frames': args.frames,
                    'development_scenes': list(development), 'held_out_scenes': held_out,
                    'selection': 'seeded random validation scenes, distinct room IDs, development rooms excluded',
                    'frame_selection': ('all views with finite camera poses' if args.frames == 0
                                        else 'position/direction farthest coverage'),
                    'sam2_points_per_side': 16, 'clip_model': 'ViT-B/32', 'clip_dim': 512,
                    'limitations': ['Automatic SAM2 masks rather than grounded VLM masks',
                                    'ViT-B/32 rather than the paper CLIP model',
                                    'RGB resized to depth resolution as in the previous benchmark']}
        if args.full_validation:
            manifest.update({'validation_scenes': validation, 'development_scenes': [], 'held_out_scenes': [],
                             'selection': 'complete official ScanNet200 validation split; all listed scans retained'})
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    labels = pd.read_csv(args.raw_root / 'scannetv2-labels.combined.tsv', sep='\t')
    sam = build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',
                     str(repository / 'checkpoints/sam2.1_hiera_large.pt'), device='cuda')
    generator = SAM2AutomaticMaskGenerator(sam, points_per_side=16, points_per_batch=64)
    adapter, preprocess = clip.load('ViT-B/32', device='cuda')
    adapter.eval()
    reader = ScanReader(str(args.output_root), 'cpu')
    with torch.inference_mode():
        scenes = manifest.get('validation_scenes', manifest['development_scenes'] + manifest['held_out_scenes'])
        for scene in scenes:
            info_path = args.output_root / f'{scene}_preparation.json'
            previous = json.loads(info_path.read_text()) if info_path.exists() else {}
            if f'frames{args.frames}' in previous and scene_cache_complete(args.output_root, scene, previous[f'frames{args.frames}']):
                print(json.dumps({'scene': scene, 'cached': True}), flush=True)
                continue
            started = time.perf_counter()
            point_path = args.output_root / 'scannet200/val' / f'{scene}.pth'
            point_path.parent.mkdir(parents=True, exist_ok=True)
            if not point_path.exists():
                handle_process(str(args.raw_root / 'scans' / scene), str(args.output_root / 'scannet200'), labels, [], [scene])
            sensor = SensorData(str(args.raw_root / 'scans' / scene / f'{scene}.sens'), load_frames=False)
            poses = np.stack([frame.camera_to_world for frame in sensor.frames])
            valid = np.flatnonzero(np.isfinite(poses).all(axis=(1, 2)))
            if not len(valid):
                raise ValueError(f'{scene} has no finite camera poses')
            selection = diverse_frames(poses, valid, args.frames)
            selected = sorted(selection)
            sensor.frames.clear()
            image_root = args.output_root / 'scannet_2d/val' / scene
            for folder in ('color', 'depth', 'pose'):
                (image_root / folder).mkdir(parents=True, exist_ok=True)
            (args.output_root / 'intrinsic').mkdir(exist_ok=True)
            np.savetxt(args.output_root / 'intrinsic' / f'{scene}.txt', sensor.intrinsic_depth)
            records = []
            for index in selected:
                frame_started = time.perf_counter()
                image_path = image_root / 'color' / f'{index}.jpg'
                cache_path = args.output_root / 'masks/val' / scene / 'sam2' / f'{index}.pth'
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                if (cache_path.exists() and image_path.exists() and
                        (image_root / 'depth' / f'{index}.png').exists() and
                        (image_root / 'pose' / f'{index}.txt').exists()):
                    records.append({'index': index, 'reused': True})
                    continue
                frame = sensor.read_frames([index])[index]
                color = frame.decompress_color(sensor.color_compression_type)
                image = Image.fromarray(color).resize((640, 480), Image.Resampling.BILINEAR)
                image.save(image_path)
                # Match inference pixels to the JPEG used by the data reader.
                image = Image.open(image_path).convert('RGB')
                depth = np.frombuffer(frame.decompress_depth(sensor.depth_compression_type), dtype='<u2').reshape(sensor.depth_height, sensor.depth_width)
                if not cv2.imwrite(str(image_root / 'depth' / f'{index}.png'), depth):
                    raise RuntimeError('depth export failed')
                np.savetxt(image_root / 'pose' / f'{index}.txt', frame.camera_to_world)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    annotations = generator.generate(np.array(image))
                annotations = [annotation for annotation in annotations if annotation['segmentation'].sum() >= 32]
                if annotations:
                    masks = torch.from_numpy(np.stack([annotation['segmentation'] for annotation in annotations]))
                    crops = torch.stack([preprocess(image.crop(Image.fromarray(annotation['segmentation']).getbbox())) for annotation in annotations])
                    features = []
                    with torch.autocast('cuda', dtype=torch.float16):
                        for batch in crops.split(64):
                            features.append(torch.nn.functional.normalize(adapter.encode_image(batch.cuda()), dim=-1).cpu())
                    features = torch.cat(features)
                    scores = torch.tensor([annotation['predicted_iou'] for annotation in annotations])
                else:
                    masks, features, scores = torch.empty(0, 480, 640, dtype=torch.bool), torch.empty(0, 512), torch.empty(0)
                reader.save_mask(str(image_path), 'val', scene, {'masks': masks, 'img_feat': features, 'conf': scores})
                torch.cuda.synchronize()
                records.append({'index': index, 'masks': len(masks), 'seconds': time.perf_counter() - frame_started})
            info = {**previous, 'scene': scene, f'frames{args.frames}': selected, 'frames': records,
                    'seconds': time.perf_counter() - started, 'sensor_total_frames': sensor.num_frames}
            temporary = info_path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(info, indent=2) + '\n')
            temporary.replace(info_path)
            print(json.dumps({'scene': scene, 'prepared_frames': len(selected), 'seconds': info['seconds']}), flush=True)
            del sensor, poses
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
