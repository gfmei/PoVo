"""Original vocabulary-free PoVo: LLaVA discovery, Grounding DINO, and SAM."""

import copy

import clip
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from libs.lib_data import get_intrinsic, resize_mask, refine_masks, DetectionResult, get_boxes
from libs.mapper import PointCloudToImageMapper
from libs.spatial_neighbors import DEFAULT_CURVE_ORDERS
from libs.instance_clustering import cluster_graph
from libs.instance_graph import construct_superpoint_adjacency_graph_radius
from libs.graph_affinity import graph_appearance_settings
from libs.inference import accumulate_mask_features, select_camera_frames, score_instances, resize_depth_and_intrinsic


class SAM2Mask(nn.Module):
    def __init__(self, cfg, global_intrinsic=None, class_names=None, device='cuda', data_reader=None,
                 image_vocabulary=None, vlm_backend=None, *args, **kwargs):
        from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

        super().__init__(*args, **kwargs)
        if image_vocabulary is not None and not callable(image_vocabulary):
            raise TypeError('image_vocabulary must be a callable returning object names.')
        self.device = device
        self.clip_adapter, self.clip_preprocess = clip.load(cfg.foundation_model.clip_model, device=device)

        self.clip_dim = cfg.foundation_model.clip_dim
        self.img_dim = cfg.data.img_dim
        self.depth_scale = cfg.data.depth_scale

        # Pointcloud Image mapper
        if global_intrinsic is None:
            global_intrinsic = get_intrinsic(self.img_dim, intrinsic=None)
        self.pointcloud_mapper = PointCloudToImageMapper(
            image_dim=self.img_dim,
            intrinsics=global_intrinsic,
            cut_bound=cfg.data.cut_num_pixel_boundary
        )

        detector_id = cfg.foundation_model.detector_id
        detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
        self.object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)
        segmenter_id = cfg.foundation_model.segmenter_id
        segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"
        self.segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
        self.processor = AutoProcessor.from_pretrained(segmenter_id)

        self.data_reader = data_reader
        self.config = cfg
        self.image_vocabulary = image_vocabulary
        self.vlm_backend = vlm_backend
        self.class_names = None if class_names is None else tuple(class_names)
        if class_names is not None:
            self.cls_emds = self.text_encoder(class_names)
        else:
            self.cls_emds = None
            if image_vocabulary is None:
                self._set_model()

    def _set_model(self):
        from models.vlm import VisionLanguageAssistant
        from models.vocabulary import DEFAULT_MODEL, load_vocabulary_model

        if self.vlm_backend is None:
            foundation = self.config.foundation_model
            self.vlm_backend = VisionLanguageAssistant(*load_vocabulary_model(
                getattr(foundation, 'vocabulary_model', DEFAULT_MODEL), self.device,
                revision=getattr(foundation, 'vocabulary_revision', None),
                cache_dir=getattr(foundation, 'vocabulary_cache_dir', None),
                load_in_4bit=getattr(foundation, 'vocabulary_load_in_4bit', True),
                local_files_only=getattr(foundation, 'vocabulary_local_files_only', False)))
        # Keep the common generation interface for image vocabulary discovery.
        self.vlm_model = self.vlm_backend.model
        self.vlm_processor = self.vlm_backend.processor

    def detect(self, image, labels, threshold: float = 0.4):
        labels = [label if label.endswith(".") else label + "." for label in labels]

        results = self.object_detector(image, candidate_labels=labels, threshold=threshold)
        results = [DetectionResult.from_dict(result) for result in results]

        return results

    def segment(self,
                image: Image.Image,
                detection_results,
                polygon_refinement
                ):
        """
        Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
        """
        boxes = get_boxes(detection_results)
        inputs = self.processor(images=image, input_boxes=boxes, return_tensors="pt").to(self.device)

        outputs = self.segmentator(**inputs)
        masks = self.processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]

        masks = refine_masks(masks, polygon_refinement)

        for detection_result, mask in zip(detection_results, masks):
            detection_result.mask = mask

        return detection_results

    def get_image_level_names(self, img):
        if getattr(self, 'image_vocabulary', None) is not None:
            return self.image_vocabulary(img)
        from models.vocabulary import discover_image_categories

        return discover_image_categories([img], self.vlm_model, self.vlm_processor)[0]

    def crop_and_extract_features(self, image, detections, thresh=32):
        cropped_tensors = []
        mask_list = []
        scores = []
        for detection in detections:
            score = detection.score
            mask = detection.mask
            if mask.sum() < thresh:
                continue
            mask_resized = resize_mask(mask, image.size)
            mask_np = mask_resized.astype(np.uint8)
            mask_pil = Image.fromarray(mask_np)
            mask_list.append(torch.from_numpy(mask_resized.astype(bool)))
            bbox = mask_pil.getbbox()
            cropped_image = copy.deepcopy(image).crop(bbox)
            cropped_tensors.append(self.clip_preprocess(cropped_image))
            scores.append(score)
        if not mask_list:
            return (
                torch.empty((0, self.clip_dim), device=self.device),
                torch.empty((0, image.height, image.width), dtype=torch.bool),
                torch.empty((0,), device=self.device),
            )
        masks = torch.stack(mask_list, dim=0)
        scores = torch.tensor(scores).to(self.device)
        imgs = torch.stack(cropped_tensors).to(self.device)
        img_batches = torch.split(imgs, 64, dim=0)
        image_features = []
        # Batch forwarding CLIP
        with torch.no_grad(), torch.autocast(device_type=torch.device(self.device).type,
                                            enabled=torch.device(self.device).type == 'cuda'):
            for img_batch in img_batches:
                image_feat = self.clip_adapter.encode_image(img_batch)
                image_feat /= image_feat.norm(dim=-1, keepdim=True)
                image_features.append(image_feat)
        image_features = torch.cat(image_features, dim=0)

        return image_features, masks, scores

    def text_encoder(self, class_names):
        if not class_names:
            return None
        with torch.no_grad(), torch.autocast(device_type=torch.device(self.device).type,
                                            enabled=torch.device(self.device).type == 'cuda'):
            text_features = self.clip_adapter.encode_text(
                clip.tokenize(["a " + label for label in class_names]).to(self.device))
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.half()


    @torch.no_grad()
    def forward(self, scene_id, split='val', class_names=None, bs=256, k_img=None):
        if class_names is None:
            class_names = getattr(self, 'class_names', None)
        config = getattr(self, 'config', None)
        clustering = getattr(config, 'clustering', None)
        graph_config = getattr(config, 'graph', None)
        appearance = graph_appearance_settings(graph_config)
        # Load point cloud data
        points, spp, ins_gt, sem_gt = self.data_reader.get_pcd(scene_id, split)
        img_dirs = self.data_reader.get_image_dirs(scene_id, split)
        n_points = points.shape[0]
        if n_points == 0:
            self.instance_objectness = torch.empty(0, device=self.device)
            self.instance_class_ids = torch.empty(0, dtype=torch.long, device=self.device)
            self.discovered_class_names = ()
            return (torch.empty((0, 0), device=self.device), torch.empty(0, device=self.device),
                    None, ins_gt.long(), sem_gt.long())

        _, spp = torch.unique(spp, return_inverse=True)
        sem_features = torch.zeros((n_points, self.clip_dim), device=self.device)
        counter = torch.zeros((n_points, 1), device=self.device)

        masks_list = []
        mapping_list = []
        score_list = []
        mask_features_list = ([] if appearance['appearance_mode'] == 'histogram_rgb'
                              and appearance['semantic_weight'] > 0 else None)

        # Limit the number of sampled RGB-D frames.
        if k_img is None:
            k_img = getattr(getattr(config, 'data', None), 'max_frames', 50)
        img_dirs = select_camera_frames(img_dirs, self.data_reader, k_img)
        num_img = len(img_dirs)
        flag = class_names is None
        class_names = [] if flag else list(class_names)
        # Process images
        for img_id, img_dir in enumerate(img_dirs):
            if img_id % 100 == 0:
                print(f'Processing image {img_id}/{num_img}')
            # Load depth and convert to float32 to reduce memory
            depth = self.data_reader.get_depth(img_dir).astype(np.float32) / self.depth_scale

            # Get pose and intrinsics
            pose = self.data_reader.get_pose(img_dir).astype(np.float32)
            if not np.isfinite(pose).all():
                continue
            translated_intrinsics = self.data_reader.get_intrinsic(scene_id)
            depth, translated_intrinsics = resize_depth_and_intrinsic(
                depth, translated_intrinsics, self.img_dim)
            if flag:
                image = self.data_reader.get_image(img_dir).resize(tuple(self.img_dim))
                cls_names = list(self.get_image_level_names(image))
                n_objects = len(cls_names)
                if n_objects < 1:
                    continue
                if n_objects > 0:
                    segment_size = 5  # scannetpp_benchmark
                    segments = [cls_names[i: i + segment_size]
                                for i in range(0, len(cls_names), segment_size)]
                class_names.extend(name for name in cls_names if name not in class_names)
                detections = []
                for cls_name in segments:
                    detection = self.detect(image, cls_name)
                    boxes = get_boxes(detection)
                    if boxes == [[]]:
                        continue
                    detection = self.segment(image, detection, True)
                    detections.extend(detection)
                clip_features, masks, scores = self.crop_and_extract_features(image, detections)
            else:
                # Get grounded data
                grounded_dict = self.data_reader.get_grounded_data(img_dir, split, scene_id)
                clip_features = grounded_dict["img_feat"]
                masks = grounded_dict["masks"]
                scores = grounded_dict["conf"]
                del grounded_dict

            # If no masks, skip
            if masks is None or len(masks) == 0:
                del depth, pose, translated_intrinsics, clip_features, masks, scores
                torch.cuda.empty_cache()
                continue

            # Project the point cloud into this frame.
            mapping = torch.ones([n_points, 4], dtype=int, device=self.device)
            mapping[:, 1:4] = self.pointcloud_mapper.compute_mapping_torch(
                pose, points, depth, intrinsic=translated_intrinsics
            )

            if mapping[:, 3].sum() == 0:
                # No visible points
                del depth, pose, translated_intrinsics, clip_features, masks, scores, mapping
                torch.cuda.empty_cache()
                continue

            # Convert masks to a boolean tensor on the device
            masks_t = torch.as_tensor(masks, dtype=torch.bool, device=self.device)

            # Check if resizing is needed
            orig_shape = masks_t.shape  # Expecting [Q, H, W] where Q=#masks
            if orig_shape[1] != self.img_dim[1] or orig_shape[2] != self.img_dim[0]:
                # Convert to float for interpolation
                masks_t = masks_t.unsqueeze(0).float()  # shape: [1, Q, H, W]
                masks_t = F.interpolate(
                    masks_t,
                    size=(self.img_dim[1], self.img_dim[0]),
                    mode='nearest'
                )
                # Convert back to boolean
                masks_t = (masks_t > 0.5).squeeze(0)  # shape: [Q, new_H, new_W]

            scores_t = torch.as_tensor(scores, device=self.device)

            masks_list.append(masks_t)
            mapping_list.append(mapping)
            score_list.append(scores_t.view(-1))
            if mask_features_list is not None:
                mask_features_list.append(torch.as_tensor(clip_features).detach().cpu())

            accumulate_mask_features(sem_features, counter, masks_t, clip_features,
                                     scores_t, mapping)
            del depth, pose, translated_intrinsics, clip_features, masks, scores, mapping
            torch.cuda.empty_cache()

        # Normalize sem_features
        counter[counter==0] = 1e-5
        sem_features = sem_features / counter
        sem_features = F.normalize(sem_features, dim=-1)

        # Construct adjacency and merge
        graph_text, point_rgb = None, None
        logit_scale = 100.
        if appearance['appearance_mode'] == 'histogram_rgb':
            if appearance['semantic_weight'] > 0 and class_names:
                graph_text = self.text_encoder(class_names)
                scale = getattr(getattr(self, 'clip_adapter', None), 'logit_scale', None)
                if scale is not None:
                    logit_scale = float(scale.detach().float().exp().clamp(max=100))
            if appearance['rgb_weight'] > 0 and hasattr(self.data_reader, 'get_point_colors'):
                point_rgb = self.data_reader.get_point_colors(scene_id, split)
        mask_tres = getattr(graph_config, 'mask_threshold', .3)
        sim_thres = getattr(graph_config, 'similarity_threshold', .8)
        graph_result = construct_superpoint_adjacency_graph_radius(
            points, spp, masks_list, mapping_list, score_list,
            min_points=0,  # minimum number of points required for a valid superpoint
            radius_threshold=getattr(graph_config, 'radius', 0.2),
            point_feas=sem_features,
            mask_thres=mask_tres,
            sim_thres = sim_thres,
            curve_window_size=getattr(graph_config, 'window_size', 32),
            curve_resolution=getattr(graph_config, 'resolution', 0.01),
            curve_chunk_size=getattr(graph_config, 'chunk_size', 8192),
            curve_orders=getattr(graph_config, 'curve_orders', DEFAULT_CURVE_ORDERS),
            return_evidence=True,
            minimum_conflict_fraction=getattr(graph_config, 'conflict_fraction', 0.),
            class_features=graph_text, point_rgb=point_rgb, mask_features_list=mask_features_list,
            logit_scale=logit_scale, **appearance,
        )
        adjacency_matrix, superpoint_confidence_scores, evidence = graph_result
        # Free lists after use
        del masks_list, mapping_list, score_list, mask_features_list, counter
        torch.cuda.empty_cache()
        merge_threshold = getattr(clustering, 'merge_threshold', 0.5)
        if not 0 < merge_threshold < 1:
            raise ValueError('clustering.merge_threshold must be between 0 and 1.')
        fine_labels, merged_labels, pred_conf = cluster_graph(
            adjacency_matrix, spp, superpoint_confidence_scores, evidence=evidence,
            merge_threshold=merge_threshold, budget=getattr(clustering, 'cluster_budget', 128),
            non_neighbor_weight=getattr(clustering, 'non_neighbor_weight', 1.))
        # Map non-consecutive IDs in merged_labels to consecutive IDs
        _, new_ids = torch.unique(merged_labels, return_inverse=True)
        # Unobserved superpoints are separate graph nodes, not mask proposals.
        supported = torch.where(pred_conf > 0)[0]
        instance = (new_ids[None, :] == supported[:, None]).half()
        pred_conf = pred_conf[supported]

        self.instance_objectness = pred_conf.detach()
        self.discovered_class_names = tuple(class_names)
        self.cls_emds = graph_text if graph_text is not None else self.text_encoder(class_names)
        scores_final, sem_pred, self.instance_class_ids = score_instances(
            sem_features, self.cls_emds, instance, pred_conf,
            adapter=getattr(self, 'clip_adapter', None), batch_size=bs, return_class_ids=True)
        masks_final = instance
        # Cleanup and return
        del adjacency_matrix, superpoint_confidence_scores, spp, points, sem_features
        torch.cuda.empty_cache()

        return masks_final, scores_final.half(), sem_pred, ins_gt.long(), sem_gt.long()
