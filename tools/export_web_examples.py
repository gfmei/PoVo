"""Export small, reproducible point-cloud examples from real PoVo predictions."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

POINT_DTYPE = np.dtype([('xyz', '<f4', 3), ('rgb', 'u1', 3), ('padding', 'u1'),
                       ('gt_instance', '<i2'), ('gt_class', '<i2'),
                       ('prediction', '<u2')])


def load_prediction(root, scene):
    path = root / scene / 'predictions.pth'
    if not path.is_file():
        path = root / f'{scene}.pth'
    return torch.load(path, map_location='cpu', weights_only=False), path


def export_scene(data_root, prediction_root, output_root, scene, max_points):
    cloud_path = data_root / 'scannet200/val' / f'{scene}.pth'
    cloud = torch.load(cloud_path, map_location='cpu', weights_only=False)
    prediction, pred_path = load_prediction(prediction_root, scene)
    coords = np.asarray(cloud['coords'], dtype=np.float32)
    assignment = prediction['assignment'].numpy()
    if assignment.shape != (len(coords),):
        raise ValueError(f'{scene}: point assignments must match the source cloud')
    if not np.isfinite(coords).all():
        raise ValueError(f'{scene}: nonfinite coordinates')
    if len(prediction['class_ids']) != len(prediction['scores']):
        raise ValueError(f'{scene}: each instance requires one class and score')
    classes = prediction['class_ids'].numpy()
    if (classes < -1).any() or (classes >= len(prediction['class_names'])).any():
        raise ValueError(f'{scene}: class ID outside the supplied vocabulary')
    if not torch.isfinite(prediction['scores']).all():
        raise ValueError(f'{scene}: nonfinite instance scores')
    if assignment.min(initial=0) < 0 or assignment.max(initial=0) > len(classes):
        raise ValueError(f'{scene}: assignment outside the instance range')
    if assignment.max(initial=0) > np.iinfo(np.uint16).max:
        raise ValueError(f'{scene}: too many instances for the web format')
    # RGB, ground truth, and prediction use identical deterministic point indices.
    indices = np.sort(np.random.default_rng(0).choice(len(coords), min(max_points, len(coords)), replace=False))
    center = (coords.max(0) + coords.min(0)) / 2
    scale = float(np.max(np.ptp(coords, axis=0)) / 2)
    if scale <= 0:
        raise ValueError(f'{scene}: degenerate cloud')
    data = np.zeros(len(indices), dtype=POINT_DTYPE)
    data['xyz'] = (coords[indices] - center) / scale
    data['rgb'] = np.rint((np.asarray(cloud['colors'])[indices] + 1) * 127.5).clip(0, 255).astype(np.uint8)
    semantic = np.asarray(cloud['labels'])[indices]
    ground = np.asarray(cloud['instance_ids'])[indices]
    foreground = (semantic >= 2) & (semantic < 200) & (ground >= 0)
    if ground[foreground].max(initial=0) >= np.iinfo(np.int16).max:
        raise ValueError(f'{scene}: GT instance ID exceeds the web format')
    data['gt_instance'] = np.where(foreground, ground + 1, 0)
    data['gt_class'] = np.where(foreground, semantic - 2, -1)
    data['prediction'] = assignment[indices]
    destination = output_root / scene
    destination.mkdir(parents=True, exist_ok=True)
    payload = data.tobytes()
    (destination / 'points.bin').write_bytes(payload)
    def instance_info(result):
        names = result['class_names']
        return [{'label': names[int(label)] if 0 <= int(label) < len(names) else 'unknown',
                 'class_id': int(label), 'confidence': float(score)}
                for label, score in zip(result['class_ids'], result['scores'])]
    metadata = {
        'scene': scene, 'point_count': len(coords), 'displayed_points': len(indices),
        'stride': POINT_DTYPE.itemsize, 'center': center.tolist(), 'scale_meters': scale,
        'class_names': prediction['class_names'], 'prediction': instance_info(prediction),
        'gt_instances': int(len(np.unique(np.asarray(cloud['instance_ids'])[
            (np.asarray(cloud['labels']) >= 2) & (np.asarray(cloud['instance_ids']) >= 0)]))),
        'points_sha256': hashlib.sha256(payload).hexdigest(),
        'source_sha256': {key: hashlib.sha256(path.read_bytes()).hexdigest()
                          for key, path in [('cloud', cloud_path), ('prediction', pred_path)]},
        'sampling': 'uniform without replacement; numpy default_rng seed 0; shared point indices',
        'ground_truth': 'foreground instances only; wall, floor, and ignored points shown in gray',
    }
    # Diagnostic uses the complete original cloud, before display sampling.
    from tools.recognize_instances import localization_targets
    targets = localization_targets({
        'assignment': prediction['assignment'], 'objectness': prediction['scores'],
        'gt_instance': torch.as_tensor(cloud['instance_ids']),
        'gt_semantic': torch.as_tensor(cloud['labels'])})
    localized = targets >= 0
    total = int(localized.sum())
    correct = int((classes[localized] == targets[localized]).sum())
    metadata['localized_class_accuracy'] = {
        'correct': correct, 'total': total, 'accuracy': correct / total if total else None,
        'definition': 'Correct class on predicted instances with ground-truth IoU >= 0.5; ground-truth regions have at least 100 points',
        'point_count': len(coords), 'ground_truth_use': 'evaluation only'}
    reranked = pred_path.parent / 'rerank.json'
    prepared = pred_path.parent / 'prepared.json'
    if reranked.is_file():
        record = json.loads(reranked.read_text())
        metadata['prediction_model'] = {key: record[key] for key in ['model', 'revision']}
    if prepared.is_file():
        metadata['prediction_frames'] = json.loads(prepared.read_text())['frames']
    encoded = pred_path.parent / 'l336.pth'
    if encoded.is_file():
        features = torch.load(encoded, map_location='cpu', weights_only=False)
        settings = features.get('metadata', {})
        metadata['prediction_recognition'] = {
            'encoder': features['backbone'], 'embedding_dim': features['embedding_dim'],
            'text_templates': settings.get('text_templates', ['a {}']),
            'focus_weight': settings.get('focus_weight', 0.),
        }
    (destination / 'scene.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    return metadata


def render_preview(output_root, scenes):
    """A shareable scientific figure of the exact points displayed by the viewer."""
    import colorsys
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    def palette(identity):
        return (colorsys.hsv_to_rgb((identity * .61803398875) % 1, .57 / .97, .97)
                if identity >= 0 else (.39, .46, .43))
    figure, axes = plt.subplots(len(scenes), 3, figsize=(12, 3.4 * len(scenes)), squeeze=False)
    figure.set_facecolor('#fbfcf9')
    for row, scene in enumerate(scenes):
        root = output_root / scene
        info = json.loads((root / 'scene.json').read_text())
        data = np.fromfile(root / 'points.bin', dtype=POINT_DTYPE)
        xyz = data['xyz']
        yaw, pitch = -2.1, .65
        depth = np.sin(yaw) * xyz[:, 0] + np.cos(yaw) * xyz[:, 1]
        x = np.cos(yaw) * xyz[:, 0] - np.sin(yaw) * xyz[:, 1]
        y = np.cos(pitch) * xyz[:, 2] - np.sin(pitch) * depth
        z = np.sin(pitch) * xyz[:, 2] + np.cos(pitch) * depth
        height = xyz[:, 2].min() + np.ptp(xyz[:, 2]) * .85
        order = np.argsort(z)
        order = order[xyz[order, 2] <= height]
        projected = np.column_stack([x, y]) / (3.5 - z[:, None])
        gt = np.array([palette(int(label)) for label in data['gt_class']])
        def class_colors(role):
            return np.array([palette(info[role][int(identity) - 1]['class_id']) if identity > 0 else palette(-1)
                             for identity in data[role]])
        for column, (title, colors) in enumerate([
                ('Original RGB', data['rgb'] / 255.), ('Ground truth classes', gt),
                ('PoVo prediction', class_colors('prediction'))]):
            axis = axes[row, column]
            axis.set_facecolor('#192c24')
            axis.scatter(projected[order, 0], projected[order, 1], c=colors[order], s=1.3, linewidths=0)
            axis.set_aspect('equal', adjustable='box')
            axis.set_xlim(projected[:, 0].min() - .03, projected[:, 0].max() + .03)
            axis.set_ylim(projected[:, 1].min() - .03, projected[:, 1].max() + .03)
            axis.set_xticks([]); axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.set_title(f'{scene} · {title}', fontsize=10, color='#182923', pad=12)
    figure.tight_layout(pad=1.5)
    buffer = io.BytesIO()
    figure.savefig(buffer, format='png', dpi=100)
    plt.close(figure)
    buffer.seek(0)
    Image.open(buffer).save(output_root / 'prediction.webp', quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--prediction-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--scenes', nargs='+', required=True)
    parser.add_argument('--max-points', type=int, default=30000)
    args = parser.parse_args()
    if args.max_points < 1:
        parser.error('max-points must be positive')
    for scene in args.scenes:
        result = export_scene(args.data_root, args.prediction_root, args.output_root, scene, args.max_points)
        print(json.dumps({'scene': scene, 'displayed_points': result['displayed_points']}))
    render_preview(args.output_root, args.scenes)


if __name__ == '__main__':
    main()
