"""Recognize PoVo instances with native ViT-L crops and selective Qwen reranking.

Run prepare, clip, rerank, then report. Ground truth is read only for evaluation;
it never determines instances, crops, candidate classes, or reranking selection.
"""

import argparse
import hashlib
import io
import json
import struct
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import clip
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from exp.eval_inst import BaseEval
from exp.scannet200_splits import (INSTANCE_CAT_SCANNET_200 as NAMES,
                                  HEAD_CATS_SCANNET_200, COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200)
from libs.inference import parse_frame_count, resize_depth_and_intrinsic
from libs.instance_recognition import (TEXT_TEMPLATES, class_text_embeddings, classify_instance_views,
                                       focus_instance, instance_class_confidence, instance_crop, parse_rerank_response,
                                       rerank_candidates, select_instance_views)
from models import sam2mask
from datasets.data_loader import SelectedScanReader

CLIP_MODEL = 'ViT-L/14@336px'


def preparation_settings(args):
    """Read the effective graph/clustering configuration and verify cached features."""
    import yaml
    from libs.spatial_neighbors import DEFAULT_CURVE_ORDERS
    from libs.graph_affinity import graph_appearance_settings

    config = yaml.safe_load(args.config.read_text()) or {}
    manifest = json.loads((args.data_root / 'manifest.json').read_text())
    foundation, data = config.get('foundation_model', {}), config.get('data', {})
    graph, clustering = config.get('graph', {}), config.get('clustering', {})
    settings = {
        'foundation_model': {'clip_model': foundation.get('clip_model', 'ViT-B/32'),
                             'clip_dim': foundation.get('clip_dim', 512)},
        'data': {'img_dim': data.get('img_dim', [640, 480]),
                 'depth_scale': data.get('depth_scale', 1000.),
                 'cut_num_pixel_boundary': data.get('cut_num_pixel_boundary', 10),
                 'max_frames': args.frames},
        'graph': {'radius': graph.get('radius', .2), 'window_size': graph.get('window_size', 32),
                  'resolution': graph.get('resolution', .01), 'chunk_size': graph.get('chunk_size', 8192),
                  'curve_orders': list(graph.get('curve_orders', DEFAULT_CURVE_ORDERS)),
                  'conflict_fraction': graph.get('conflict_fraction', 0.),
                  'mask_threshold': graph.get('mask_threshold', .3),
                  'similarity_threshold': graph.get('similarity_threshold', .8),
                  **graph_appearance_settings(graph)},
        'clustering': {'cluster_budget': clustering.get('cluster_budget', 128),
                       'merge_threshold': clustering.get('merge_threshold', .5),
                       'non_neighbor_weight': clustering.get('non_neighbor_weight', 1.)},
    }
    for key in ('clip_model', 'clip_dim'):
        if manifest.get(key) != settings['foundation_model'][key]:
            raise ValueError(f'configured {key} does not match cached mask features; prepare a matching cache')
    dimensions = settings['data']['img_dim']
    if len(dimensions) != 2 or any(not isinstance(value, int) or value <= 0 for value in dimensions):
        raise ValueError('data.img_dim must contain two positive integer dimensions')
    if settings['data']['depth_scale'] <= 0:
        raise ValueError('data.depth_scale must be positive')
    return settings


def preparation_metadata(args, scene, settings, selected):
    """Fingerprint all geometry/crop inputs before accepting a completed scene."""
    source = Path(__file__).resolve().parents[1]
    source_files = [Path(__file__), *[source / name for name in (
        'models/sam2mask.py', 'libs/instance_graph.py', 'libs/graph_affinity.py', 'libs/instance_clustering.py',
        'libs/lib_cluster.py', 'libs/spatial_neighbors.py', 'libs/inference.py',
        'libs/mapper.py', 'libs/instance_recognition.py', 'datasets/data_loader.py')]]
    input_files = [args.data_root / 'scannet200/val' / f'{scene}.pth',
                   args.data_root / 'intrinsic' / f'{scene}.txt',
                   args.data_root / f'{scene}_preparation.json']
    for frame in sorted(selected):
        for folder, suffix in [('color', 'jpg'), ('depth', 'png'), ('pose', 'txt')]:
            input_files.append(args.data_root / 'scannet_2d/val' / scene / folder / f'{frame}.{suffix}')
        candidates = [args.data_root / folder / 'val' / scene / 'sam2' / f'{frame}.pth'
                      for folder in ('masks', 'mask_sam2')]
        input_files.append(next((path for path in candidates if path.exists()), candidates[0]))

    def digest(path):
        result = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                result.update(block)
        return result.hexdigest()

    native = {}
    size = read_selected_colors(args.raw_root / 'scans' / scene / f'{scene}.sens', selected,
                                lambda frame, color: native.update({str(frame): hashlib.sha256(color).hexdigest()}))
    return {'settings': settings, 'selected_frames': sorted(selected),
            'native_rgb': {'size': list(size), 'selected_jpeg_sha256': native},
            'source_sha256': {str(path.relative_to(source)): digest(path) for path in source_files},
            'input_sha256': {str(path.relative_to(args.data_root)): digest(path) for path in input_files}}


def validate_preparation_cache(prepared, metadata, mask_sha256):
    if prepared.get('metadata') != metadata or prepared.get('mask_sha256') != mask_sha256:
        raise ValueError('existing preparation uses different settings or inputs; choose a fresh output root')


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_tensor(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def read_selected_colors(sensor_path, selected, consume):
    """Stream selected JPEGs to a consumer without loading whole RGB-D scans."""
    selected = set(selected)
    with sensor_path.open('rb') as handle:
        version, name_length = struct.unpack('<IQ', handle.read(12))
        if version != 4:
            raise ValueError('unsupported ScanNet sensor version')
        handle.seek(name_length + 4 * 16 * 4, 1)
        color_type, depth_type, width, height, dw, dh, depth_scale, count = struct.unpack('<iiIIIIfQ', handle.read(36))
        if color_type != 2:
            raise ValueError('native JPEG frames are required')
        if any(index < 0 or index >= count for index in selected):
            raise ValueError('selected frame is outside the sensor sequence')
        for index in range(count):
            handle.seek(80, 1)  # Pose and two timestamps.
            color_size, depth_size = struct.unpack('<QQ', handle.read(16))
            if index in selected:
                color = handle.read(color_size)
                if len(color) != color_size:
                    raise EOFError('truncated sensor color frame')
                consume(index, color)
                handle.seek(depth_size, 1)
            else:
                handle.seek(color_size + depth_size, 1)
    return width, height


def export_selected_colors(sensor_path, selected, destination):
    destination.mkdir(parents=True, exist_ok=True)
    return read_selected_colors(sensor_path, selected,
                                lambda frame, color: (destination / f'{frame}.jpg').write_bytes(color))


def freeze_scene(model, scene):
    """Retain point-feature fallback and objectness without changing masks."""
    captured = {}
    original = sam2mask.score_instances

    def capture(features, text, masks, objectness, adapter=None, batch_size=256, return_class_ids=False):
        captured['objectness'] = objectness.detach().cpu()
        known = features.norm(dim=-1) > 0
        probabilities = (adapter.logit_scale.float().exp().clamp(max=100) *
                         (F.normalize(features.float(), dim=-1) @ F.normalize(text.float(), dim=-1).T)).softmax(-1)
        probabilities *= known[:, None]
        membership = masks.float() * known[None, :]
        captured['point_probabilities'] = (membership @ probabilities / membership.sum(1, keepdim=True).clamp_min(1)).cpu()
        return original(features, text, masks, objectness, adapter, batch_size, return_class_ids)

    sam2mask.score_instances = capture
    try:
        masks, scores, semantic, gt_instance, gt_semantic = model(scene, class_names=NAMES)
    finally:
        sam2mask.score_instances = original
    keep = masks.sum(1) >= 100
    masks = masks[keep].bool()
    assignment = torch.zeros(masks.shape[1], dtype=torch.int32, device=masks.device)
    if len(masks):
        assignment = (masks.int().argmax(0) + 1) * masks.any(0)
    captured.update({'assignment': assignment.cpu(), 'gt_instance': gt_instance.cpu(), 'gt_semantic': gt_semantic.cpu()})
    captured['objectness'] = captured['objectness'][keep.cpu()]
    captured['point_probabilities'] = captured['point_probabilities'][keep.cpu()]
    captured['mask_sha256'] = hashlib.sha256(captured['assignment'].numpy().tobytes()).hexdigest()
    return captured, masks


def projected_views(model, reader, scene, masks, mask_root=None):
    points = reader.get_pcd(scene, 'val')[0]
    candidates = [[] for _ in masks]
    for image_path in reader.get_image_dirs(scene, 'val'):
        pose = reader.get_pose(image_path).astype(np.float32)
        if not np.isfinite(pose).all():
            continue
        width, height = model.img_dim
        depth, intrinsic = resize_depth_and_intrinsic(reader.get_depth(image_path).astype(np.float32) / model.depth_scale,
                                                      reader.get_intrinsic(scene), model.img_dim)
        mapping = model.pointcloud_mapper.compute_mapping_torch(pose, points, depth, intrinsic=intrinsic)
        visible = mapping[:, 2].bool()
        if not visible.any():
            continue
        pixels = mapping[visible, :2]
        membership = masks[:, visible].float()
        counts = membership.sum(1)
        cached = reader.get_grounded_data(image_path, 'val', scene)
        sam_masks = (torch.as_tensor(cached['masks'], device=points.device, dtype=torch.bool)
                     if cached['masks'] is not None else torch.empty((0, height, width), device=points.device, dtype=torch.bool))
        if len(sam_masks) and sam_masks.shape[-2:] != (height, width):
            sam_masks = F.interpolate(sam_masks[None].float(), (height, width), mode='nearest')[0].bool()
        if len(sam_masks):
            sample = sam_masks[:, pixels[:, 0], pixels[:, 1]].float()
            intersection = membership @ sample.T
            union = counts[:, None] + sample.sum(1)[None] - intersection
            iou = intersection / union.clamp_min(1)
            best_iou, best_mask = iou.max(1)
        camera = np.concatenate((pose[:3, 3] / .5, pose[:3, 2] * 2)).tolist()
        for index in torch.where(counts >= 10)[0].tolist():
            location = pixels[membership[index].bool()].cpu().numpy()
            y0, x0 = np.quantile(location, .01, axis=0)
            y1, x1 = np.quantile(location, .99, axis=0)
            box = [int(x0), int(y0), int(x1) + 1, int(y1) + 1]
            purity, mask_confidence = 1., 1.
            target_mask = None
            if len(sam_masks):
                match = int(best_mask[index])
                overlap = float(intersection[index, match])
                recall = overlap / max(float(counts[index]), 1.)
                purity = overlap / max(float(sample[match].sum()), 1.)
                if float(best_iou[index]) >= .1 and recall >= .5 and purity >= .5:
                    mask_image = Image.fromarray(sam_masks[match].cpu().numpy())
                    box = list(mask_image.getbbox())
                    mask_confidence = float(cached['conf'][match])
                    target_mask = mask_image
            if min(box[2] - box[0], box[3] - box[1]) < 8:
                continue
            coverage = float(counts[index]) / int(masks[index].sum())
            area = (box[2] - box[0]) * (box[3] - box[1]) / (width * height)
            quality = np.sqrt(coverage * area) * (.5 + .5 * purity) * mask_confidence
            view = {'frame': int(Path(image_path).stem), 'box': box,
                    'coverage': coverage, 'purity': purity, 'quality': float(quality), 'camera': camera}
            if mask_root is not None:
                if target_mask is None:
                    foreground = np.zeros((height, width), dtype=np.uint8)
                    foreground[location[:, 0], location[:, 1]] = 255
                    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
                    target_mask = Image.fromarray(foreground)
                buffer = io.BytesIO()
                target_mask.save(buffer, format='PNG')
                view['_target_png'] = buffer.getvalue()
            candidates[index].append(view)
        del cached, sam_masks, mapping, membership
    selected = [select_instance_views(views, 3) for views in candidates]
    if mask_root is not None:
        mask_root = Path(mask_root)
        mask_root.mkdir(parents=True, exist_ok=True)
        for index, instance in enumerate(selected):
            for view in instance:
                path = mask_root / f"{index}_{view['frame']}.png"
                path.write_bytes(view.pop('_target_png'))
                view['target_mask'] = str(Path(mask_root.name) / path.name)
    return selected


def prepare(args, scenes):
    reader = SelectedScanReader(str(args.data_root), 'cuda')
    settings = preparation_settings(args)
    cfg = SimpleNamespace(**{key: SimpleNamespace(**value) for key, value in settings.items()})
    model = None
    for scene in scenes:
        root = args.output_root / scene
        root.mkdir(parents=True, exist_ok=True)
        selected = json.loads((args.data_root / f'{scene}_preparation.json').read_text())[f'frames{args.frames}']
        metadata = preparation_metadata(args, scene, settings, selected)
        if all((root / name).is_file() for name in ('prepared.json', 'views.json', 'frozen.pth')):
            prepared = json.loads((root / 'prepared.json').read_text())
            frozen = torch.load(root / 'frozen.pth', map_location='cpu', weights_only=False)
            if hashlib.sha256(frozen['assignment'].numpy().tobytes()).hexdigest() != frozen['mask_sha256']:
                raise ValueError('cached instance assignment does not match its mask hash')
            validate_preparation_cache(prepared, metadata, frozen['mask_sha256'])
            print(json.dumps({'phase': 'prepare', 'scene': scene, 'cached': True}), flush=True)
            continue
        started = time.perf_counter()
        if model is None:
            model = sam2mask.SAM2Mask(cfg, data_reader=reader).eval()
        reader.selected = set(selected)
        frozen, masks = freeze_scene(model, scene)
        views = projected_views(model, reader, scene, masks, root / 'target_masks')
        selected = {view['frame'] for instance in views for view in instance}
        size = export_selected_colors(args.raw_root / 'scans' / scene / f'{scene}.sens', selected, root / 'rgb')
        for instance in views:
            for view in instance:
                native_box = [view['box'][0] * size[0] / cfg.data.img_dim[0], view['box'][1] * size[1] / cfg.data.img_dim[1],
                              view['box'][2] * size[0] / cfg.data.img_dim[0], view['box'][3] * size[1] / cfg.data.img_dim[1]]
                view['native_box'] = native_box
                with Image.open(root / 'rgb' / f"{view['frame']}.jpg") as image:
                    crop = instance_crop(image.convert('RGB'), native_box, 1.2)
                    gray = np.array(crop.convert('L').resize((128, 128)))
                    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                view['sharpness'] = sharpness
                view['weight'] = view['quality'] * (.5 + .5 * min(sharpness / 100., 1.))
        save_tensor(root / 'frozen.pth', frozen)
        save_json(root / 'views.json', views)
        save_json(root / 'prepared.json', {'scene': scene, 'frames': args.frames, 'instances': len(masks), 'metadata': metadata,
                                          'mask_sha256': frozen['mask_sha256'], 'native_size': size,
                                          'supported_instances': sum(bool(view) for view in views),
                                          'seconds': time.perf_counter() - started})
        print((root / 'prepared.json').read_text(), flush=True)
        del masks, frozen
        torch.cuda.empty_cache()


def crop_recognition_metadata(root, views):
    """Changes to target masks, pixels, prompts, or code invalidate CLIP evidence."""
    source = Path(__file__).resolve().parents[1]
    images = {root / 'rgb' / f"{view['frame']}.jpg" for instance in views for view in instance}
    masks = set()
    for instance in views:
        for view in instance:
            if 'target_mask' not in view:
                raise ValueError('prepared views lack target masks; prepare a fresh output root')
            masks.add(root / view['target_mask'])
    inputs = [root / 'frozen.pth', root / 'views.json', *sorted(images), *sorted(masks)]
    return {'backbone': CLIP_MODEL, 'vocabulary': list(NAMES), 'text_templates': list(TEXT_TEMPLATES),
            'focus_weight': .5, 'retained_context': .25, 'crop_scales': [1., 1.2, 1.5],
            'input_sha256': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in inputs},
            'source_sha256': {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in [Path(__file__), source / 'libs/instance_recognition.py']}}


def extract(args, scenes, backbone=None):
    """Encode crops, optionally reusing an explicitly supplied CLIP backbone."""
    model, preprocess = backbone if backbone is not None else (None, None)
    text = None
    for scene in scenes:
        root = args.output_root / scene
        target = root / 'l336.pth'
        views = json.loads((root / 'views.json').read_text())
        metadata = crop_recognition_metadata(root, views)
        if target.exists():
            cached = torch.load(target, map_location='cpu', weights_only=False)
            if cached.get('metadata') != metadata:
                raise ValueError('existing CLIP evidence uses different settings or inputs; choose a fresh output root')
            print(json.dumps({'phase': 'clip', 'scene': scene, 'cached': True}), flush=True)
            continue
        if text is None:
            if model is None:
                model, preprocess = clip.load(CLIP_MODEL, device='cuda')
            if getattr(getattr(model, 'visual', None), 'input_resolution', None) != 336:
                raise ValueError('crop recognition requires the matched CLIP ViT-L/14@336px backbone')
            model.eval()
            text = class_text_embeddings(model, NAMES, clip.tokenize)
            if text.shape != (len(NAMES), 768):
                raise ValueError('crop recognition requires 768-dimensional matched class embeddings')
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        tensors, owners, weights = [], [], []
        for index, instance in enumerate(views):
            for view in instance:
                with Image.open(root / 'rgb' / f"{view['frame']}.jpg") as image:
                    image = image.convert('RGB')
                    with Image.open(root / view['target_mask']) as foreground:
                        focused = focus_instance(image, foreground)
                    for visual in (image, focused):
                        for expansion in (1., 1.2, 1.5):
                            tensors.append(preprocess(instance_crop(visual, view['native_box'], expansion)))
                            owners.append(index)
                            weights.append(view['weight'] / 6)
        embeddings = []
        for start in range(0, len(tensors), args.batch_size):
            batch = torch.stack(tensors[start:start + args.batch_size]).cuda()
            embeddings.append(model.encode_image(batch).float())
        embeddings = torch.cat(embeddings) if embeddings else text.new_empty((0, text.shape[-1]))
        probabilities, similarities, supported, features = classify_instance_views(
            embeddings, torch.tensor(owners, device='cuda', dtype=torch.long), torch.tensor(weights, device='cuda'),
            text, model.logit_scale.exp().clamp(max=100).item(), len(views), return_features=True)
        torch.cuda.synchronize()
        save_tensor(target, {'probabilities': probabilities.cpu(), 'similarities': similarities.cpu(), 'supported': supported.cpu(),
                    'seconds': time.perf_counter() - started, 'peak_gpu_gib': torch.cuda.max_memory_allocated() / 2**30,
                    'backbone': CLIP_MODEL, 'embedding_dim': text.shape[-1], 'crop_count': len(tensors),
                    'features': features.cpu(), 'text_embeddings': text.cpu(), 'metadata': metadata,
                    'mask_sha256': json.loads((root / 'prepared.json').read_text())['mask_sha256']})
        print(json.dumps({'phase': 'clip', 'scene': scene, 'crops': len(tensors), 'seconds': time.perf_counter() - started}), flush=True)
        del tensors, embeddings
    del model, text
    torch.cuda.empty_cache()


RERANK_PROMPT = (
    'These crops show the same target object from different views. Identify the main centered object, '
    'using its visible shape and function. Choose exactly one label from: {names}. '
    'Use unknown if the object is unclear or none fits. Return only JSON: {{"label":"chosen label"}}.')


def validate_rerank_cache(cached, metadata, mask_sha256):
    """A changed checkpoint, prompt, crop, or candidate set requires fresh results."""
    if cached.get('metadata') != metadata or cached.get('mask_sha256') != mask_sha256:
        raise ValueError('existing reranking uses different settings or inputs; choose a fresh output root')


def rerank(args, scenes, assistant=None):
    from models.vlm import DEFAULT_MODEL, DEFAULT_REVISION, VisionLanguageAssistant
    import yaml

    config = yaml.safe_load(args.config.read_text())['foundation_model']
    if args.vlm_cache_dir is not None:
        config['reranker_cache_dir'] = str(args.vlm_cache_dir)
    if args.local_files_only:
        config['reranker_local_files_only'] = True
    model_id = config.get('reranker_model', DEFAULT_MODEL)
    revision = config.get('reranker_revision', DEFAULT_REVISION if model_id == DEFAULT_MODEL else None)
    if revision is None:
        raise ValueError('pin foundation_model.reranker_revision for reproducible recognition')
    source = Path(__file__).resolve().parents[1]
    settings = {
        'model': model_id, 'revision': revision,
        'load_in_4bit': config.get('reranker_load_in_4bit', True),
        'max_new_tokens': 64, 'thinking': False, 'do_sample': False,
        'repetition_penalty': 1.05, 'no_repeat_ngram_size': 6,
        'prompt': RERANK_PROMPT, 'views': 2, 'crop_scale': 1.2, 'choice_seed': 20261010,
        'selection': {'max_probability': .55, 'cosine_margin': .015, 'top_k': 5},
        'source_sha256': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), source / 'models/vlm.py', source / 'libs/instance_recognition.py']},
    }
    for scene in scenes:
        root = args.output_root / scene
        target = root / 'rerank.json'
        data = torch.load(root / 'l336.pth', weights_only=False)
        views = json.loads((root / 'views.json').read_text())
        selected, candidates = rerank_candidates(data['probabilities'], data['similarities'], data['supported'])
        selected = [index for index in selected if views[index]]
        rgb_files = sorted({root / 'rgb' / f"{view['frame']}.jpg"
                            for index in selected for view in views[index][:2]})
        metadata = dict(settings, input_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [root / 'l336.pth', root / 'views.json', *rgb_files]})
        if target.exists():
            validate_rerank_cache(json.loads(target.read_text()), metadata, data['mask_sha256'])
            print(json.dumps({'phase': 'rerank', 'scene': scene, 'cached': True}), flush=True)
            continue
        # Load only when an uncached scene actually needs the language model.
        if selected and assistant is None:
            assistant = VisionLanguageAssistant.from_config(SimpleNamespace(**config))
        records = []
        started = time.perf_counter()
        for index in selected:
            names = [NAMES[i] for i in candidates[index].tolist()]
            np.random.default_rng(index + 20261010).shuffle(names)
            crops = []
            for view in views[index][:2]:
                with Image.open(root / 'rgb' / f"{view['frame']}.jpg") as image:
                    crops.append(instance_crop(image.convert('RGB'), view['native_box'], 1.2))
            response = assistant.generate(crops, RERANK_PROMPT.format(names=', '.join(names)), max_new_tokens=64)
            label = parse_rerank_response(response, names)
            records.append({'instance': index, 'candidates': names, 'response': response,
                            'label': label, 'class_id': NAMES.index(label) if label is not None else None})
        save_json(target, {'model': model_id, 'revision': revision, 'metadata': metadata,
                           'selection': settings['selection'],
                           'confidence': 'selected-class CLIP probability times objectness',
                           'records': records, 'seconds': time.perf_counter() - started,
                           'mask_sha256': data['mask_sha256']})
        print(json.dumps({'phase': 'rerank', 'scene': scene, 'instances': len(records),
                          'seconds': time.perf_counter() - started}), flush=True)


def localization_targets(frozen):
    """Diagnostic targets on predicted masks with IoU >= .5, never for inference."""
    assignment = frozen['assignment'].numpy()
    instance = frozen['gt_instance'].numpy()
    semantic = frozen['gt_semantic'].numpy()
    targets = np.full(len(frozen['objectness']), -1, dtype=np.int64)
    best_iou = np.zeros(len(targets))
    sizes = np.bincount(assignment, minlength=len(targets) + 1)[1:]
    for identity in np.unique(instance[instance >= 0]):
        mask = instance == identity
        labels = semantic[mask]
        labels = labels[(labels >= 2) & (labels < len(NAMES) + 2)]
        if mask.sum() < 100 or not len(labels):
            continue
        label = np.bincount(labels).argmax() - 2
        intersections = np.bincount(assignment[mask], minlength=len(targets) + 1)[1:]
        ious = intersections / np.maximum(sizes + mask.sum() - intersections, 1)
        better = (ious >= .5) & (ious > best_iou)
        best_iou[better], targets[better] = ious[better], label
    return targets


def evaluation_metrics(evaluator, ap, recall):
    averages = evaluator.compute_averages(ap, recall)
    return {key: float(averages[key]) if np.isfinite(averages[key]) else None
            for key in ('all_ap', 'all_ap_50%', 'all_ap_25%')} | {
        'evaluated_classes': int(np.isfinite(ap[0, :, 0]).sum())}


def detailed_metrics(evaluator, ap, recall):
    averages = evaluator.compute_averages(ap, recall)
    classes = {name: {key: float(value) if np.isfinite(value) else None for key, value in metrics.items()}
               for name, metrics in averages['classes'].items()}
    groups = {}
    for group, names in [('head', HEAD_CATS_SCANNET_200), ('common', COMMON_CATS_SCANNET_200), ('tail', TAIL_CATS_SCANNET_200)]:
        eligible = [classes[name] for name in names if name in classes and classes[name]['ap'] is not None]
        groups[group] = {'evaluated_classes': len(eligible),
                         **{key: float(np.mean([entry[key] for entry in eligible])) if eligible else None
                            for key in ('ap', 'ap50%', 'ap25%')}}
    return {'per_class': classes, 'frequency_groups': groups}


def instance_predictions(frozen, clip_data, reranked):
    """Use the chosen class's probability, retaining point evidence for absent crops."""
    if clip_data['mask_sha256'] != frozen['mask_sha256'] or reranked['mask_sha256'] != frozen['mask_sha256']:
        raise ValueError('recognition artifacts do not match the instance masks')
    probabilities = clip_data['probabilities'].clone()
    unsupported = ~clip_data['supported']
    probabilities[unsupported] = frozen['point_probabilities'][unsupported]
    classes = probabilities.argmax(1).masked_fill(probabilities.sum(1) <= 0, -1)
    for record in reranked['records']:
        label = record['class_id']
        if label is not None:
            classes[record['instance']] = label
    confidence = instance_class_confidence(probabilities, classes, frozen['objectness'])
    return classes, confidence, probabilities


def localization_predictions(scene, frozen):
    """Evaluate instance separation using objectness, including unknown labels."""
    assignment = frozen['assignment'].numpy()
    return [{'scan_id': scene, 'label_id': 1, 'conf': float(score),
             'pred_mask': (assignment == index + 1).astype(np.uint8)}
            for index, score in enumerate(frozen['objectness']) if score > 0]


def report(args, scenes):
    """Export and evaluate only the retained recognition workflow."""
    evaluator = BaseEval(NAMES, use_label=True, dataset_name='scannet200')
    localization_evaluator = BaseEval(NAMES, use_label=False, dataset_name='scannet200')
    matches, localization_matches = {}, {}
    counts = {'localized': 0, 'top1_correct': 0, 'top5_correct': 0}
    confusion = {}
    summary = {'scenes': scenes, 'frames': args.frames,
               'scope': 'Native ViT-L crops and selective Qwen reranking on PoVo masks',
               'confidence': 'selected-class CLIP probability times objectness',
               'averaging': 'AP excludes classes without eligible ground truth',
               'limitations': ['Automatic SAM2 masks; supplied 198-class vocabulary',
                               'Graph configuration selected on 20 validation rooms; recognition selected on 12 validation rooms',
                               'Recognition confirmed on 40 other validation rooms; not an independent test split'],
               'per_scene': {}, 'aggregate': {},
               'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [Path(__file__), Path(__file__).resolve().parents[1] / 'libs/instance_recognition.py']}}
    for scene in scenes:
        root = args.output_root / scene
        frozen = torch.load(root / 'frozen.pth', weights_only=False)
        data = torch.load(root / 'l336.pth', weights_only=False)
        reranked = json.loads((root / 'rerank.json').read_text())
        identity = {'model': reranked['model'], 'revision': reranked['revision']}
        if 'reranker' in summary and summary['reranker'] != identity:
            raise ValueError('a report cannot combine different reranker checkpoints')
        summary['reranker'] = identity
        protocols = {
            'clip_protocol': {key: value for key, value in data.get('metadata', {}).items() if key != 'input_sha256'},
            'reranking_protocol': {key: value for key, value in reranked.get('metadata', {}).items() if key != 'input_sha256'},
        }
        for key, protocol in protocols.items():
            if key in summary and summary[key] != protocol:
                raise ValueError('a report cannot combine different recognition protocols')
            summary[key] = protocol
        classes, confidence, probabilities = instance_predictions(frozen, data, reranked)
        save_tensor(root / 'predictions.pth', {'assignment': frozen['assignment'], 'class_ids': classes, 'scores': confidence,
                                             'class_names': list(NAMES), 'mask_sha256': frozen['mask_sha256']})
        predictions = [{'scan_id': scene, 'label_id': int(label) + 1, 'conf': float(confidence[index]),
                        'pred_mask': (frozen['assignment'].numpy() == index + 1).astype(np.uint8)}
                       for index, label in enumerate(classes) if confidence[index] > 0]
        gt, pred = evaluator.assign_instances_for_scan(predictions, frozen['gt_semantic'].numpy(), frozen['gt_instance'].numpy())
        matches[scene] = {'gt': gt, 'pred': pred}
        gt_local, pred_local = localization_evaluator.assign_instances_for_scan(
            localization_predictions(scene, frozen), frozen['gt_semantic'].numpy(), frozen['gt_instance'].numpy())
        localization_matches[scene] = {'gt': gt_local, 'pred': pred_local}
        ap, recall = evaluator.evaluate_matches({scene: matches[scene]})
        targets = localization_targets(frozen)
        localized = targets >= 0
        top5 = probabilities.topk(5, dim=-1).indices.numpy()
        class_ids = classes.numpy()
        correct = class_ids[localized] == targets[localized]
        correct5 = ((top5[localized] == targets[localized, None]).any(1) &
                    (probabilities.sum(1).numpy()[localized] > 0))
        counts['localized'] += int(localized.sum())
        counts['top1_correct'] += int(correct.sum())
        counts['top5_correct'] += int(correct5.sum())
        for expected, predicted in zip(targets[localized], class_ids[localized]):
            if expected != predicted:
                key = NAMES[int(expected)] + ' -> ' + (NAMES[int(predicted)] if predicted >= 0 else 'unknown')
                confusion[key] = confusion.get(key, 0) + 1
        summary['per_scene'][scene] = {'instances': len(targets), 'localized': int(localized.sum()),
                                     'top1_correct': int(correct.sum()),
                                     'top1_accuracy_localized': float(correct.mean()) if len(correct) else None,
                                     'mask_sha256': frozen['mask_sha256'], 'metrics': evaluation_metrics(evaluator, ap, recall),
                                     'timing_seconds': {'clip': data['seconds'], 'rerank': reranked['seconds']}}
        print(json.dumps({'phase': 'report', 'scene': scene, 'completed': len(matches), 'total': len(scenes)}), flush=True)
    if matches:
        ap, recall = evaluator.evaluate_matches(matches)
        total = counts['localized']
        summary['aggregate'] = {**evaluation_metrics(evaluator, ap, recall), **counts,
                                'top1_accuracy_localized': counts['top1_correct'] / total if total else None,
                                'top5_accuracy_localized': counts['top5_correct'] / total if total else None,
                                'confusions': dict(sorted(confusion.items(), key=lambda item: (-item[1], item[0])))}
        summary.update(detailed_metrics(evaluator, ap, recall))
        ap_local, recall_local = localization_evaluator.evaluate_matches(localization_matches)
        summary['class_agnostic'] = {**evaluation_metrics(localization_evaluator, ap_local, recall_local),
                                     'confidence': 'instance objectness; recognition labels are ignored'}
        summary['scene_count'] = len(matches)
        summary['instance_count'] = sum(value['instances'] for value in summary['per_scene'].values())
    save_json(args.output_root / f'{args.cohort}_summary.json', summary)
    print(json.dumps(summary['aggregate'], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', nargs='?', default='run', choices=['run', 'prepare', 'clip', 'rerank', 'report'])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--cohort', choices=['development', 'held_out', 'validation'], required=True)
    parser.add_argument('--frames', type=parse_frame_count, default=50,
                        help='views per scene (default: 50); all uses the complete prepared view set')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/scannet200.yaml')
    parser.add_argument('--vlm-cache-dir', type=Path)
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data_root / 'manifest.json').read_text())
    key = args.cohort + '_scenes'
    if key not in manifest or not manifest[key]:
        parser.error('requested cohort is absent from the preparation manifest')
    if manifest['target_frames'] != args.frames:
        parser.error('frame count must match the preparation manifest')
    scenes = manifest[key]
    for scene in scenes:
        prepared = args.output_root / scene / 'prepared.json'
        if prepared.exists() and json.loads(prepared.read_text())['frames'] != args.frames:
            parser.error('existing predictions use a different frame count; choose a fresh output root')
    if args.batch_size < 1:
        parser.error('batch count must be positive')
    phases = {'prepare': prepare, 'clip': extract, 'rerank': rerank, 'report': report}
    with torch.inference_mode():
        selected = list(phases) if args.phase == 'run' else [args.phase]
        for phase in selected:
            phases[phase](args, scenes)


if __name__ == '__main__':
    main()
