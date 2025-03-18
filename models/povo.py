import copy
import random

import clip
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

from libs.lib_cluster import merge_superpoints_based_on_adj_matrix, perform_clustering_and_merge_clusters
from libs.lib_data import get_intrinsic, num_to_natural_torch, resize_mask, refine_masks, DetectionResult, get_boxes
from libs.mapper import PointCloudToImageMapper
from models.idefics_utils import load_idefics_model, get_image_category_from_idefics
from models.llava_utils import load_llava_model_hg, get_image_category_from_llavahg


def farthest_point_sampling(pts, sample_k):
    """
    Performs Farthest Point Sampling on pts (N, 3) to pick sample_k points.
    Returns indices of the chosen points, shape: (sample_k,).

    Naive O(N * sample_k) approach using iterative distance updates.
    If N <= sample_k, simply returns all indices.

    Args:
        pts (torch.Tensor): (N, 3) set of points.
        sample_k (int): Number of samples to pick.

    Returns:
        selected_idx (torch.LongTensor): (sample_k,) farthest point sample indices.
    """
    device = pts.device
    N = pts.shape[0]

    # If there are fewer than sample_k points, return them all
    if N <= sample_k:
        return torch.arange(N, device=device, dtype=torch.long)

    # Initialize an array to store the minimum distance to any selected point
    distances = torch.full((N,), float('inf'), device=device)

    # Pick an initial point randomly
    first_idx = torch.randint(0, N, (1,), device=device)
    selected_idx = [first_idx.item()]

    # Update distances to the first selected point
    dist_to_first = torch.norm(pts - pts[first_idx], dim=1)
    distances = torch.min(distances, dist_to_first)

    # Iteratively pick next farthest point
    for _ in range(1, sample_k):
        # farthest point is one with max distance from the current set
        farthest_idx = torch.argmax(distances)
        selected_idx.append(farthest_idx.item())

        # Update the distances (minimum distance to any selected point)
        dist_to_new = torch.norm(pts - pts[farthest_idx], dim=1)
        distances = torch.min(distances, dist_to_new)

    return torch.tensor(selected_idx, device=device, dtype=torch.long)


def construct_superpoint_adjacency_graph_radius(
        pcds,  # (N, 3) tensor of point coordinates.
        spt_labels,  # (N,) tensor of superpoint labels (0,1,...).
        masks_list,  # List of masks for each frame, shape (num_masks, H, W) per frame.
        mapping_list,  # List of mappings for each frame, shape (N, 4) (visibility, y, x, z).
        sam_scores_list,  # List of lists of confidence scores for each frame.
        sample_k=10,  # Number of representative points per superpoint.
        min_points=5,  # Minimum points for a valid superpoint.
        radius_threshold=0.2,  # Radius threshold to connect two superpoints if min-dist < this value.
        point_feas=None,  # (N, F) optional tensor of point features.
        mask_thres=0.2,  # Threshold for mask contribution.
        sim_thres=0.5,  # Threshold for feature similarity.
        n_mask=5,  # Maximum number of masks to consider per superpoint pair.
        top_k=10,  # For each mask, only keep top_k superpoints by coverage fraction.
        batch_size=32  # Batch size for chunking distance computations.
):
    """
    Constructs a superpoint adjacency graph using a radius threshold, with:
      - Farthest Point Sampling (FPS) for representatives.
      - For each mask, only keep the top_k superpoints by coverage fraction.

    1. Identify valid superpoints (>= min_points).
    2. Sample up to sample_k points from each superpoint using FPS.
    3. Compute pairwise min-distances between superpoints' reps -> radius-based adjacency.
    4. Aggregate mask coverage. For each mask, keep only top_k superpoints (zero out others).
    5. Build mask_contribution using mask_thres.
    6. (Optional) compute features -> modulate adjacency by overlap & feature sim.
    7. Compute superpoint confidence scores from coverage fractions.

    Returns:
        final_adjacency (torch.Tensor): (M, M) adjacency among valid superpoints.
        superpoint_confidence_scores (torch.Tensor): (M,) confidence scores for valid superpoints.
    """
    device = pcds.device
    spt_labels = spt_labels.to(device).long()

    # ================ STEP 1: Identify valid superpoints ================
    unique_spts, sp_counts = torch.unique(spt_labels, return_counts=True)
    valid_mask = (sp_counts >= min_points)
    if not valid_mask.any():
        raise ValueError(f"No superpoints have at least {min_points} points.")

    valid_superpoints = unique_spts[valid_mask]  # shape: [M]
    M = valid_superpoints.size(0)

    # ================ STEP 2: Representative sampling (FPS) ================
    reps_list = []
    for sp_id in valid_superpoints:
        idx = torch.nonzero(spt_labels == sp_id, as_tuple=False).view(-1)
        n_points_sp = idx.numel()
        if n_points_sp > sample_k:
            # perm = torch.randperm(n_points_sp, device=device)[:sample_k]
            # rep_idx = idx[perm]
            # Gather all points in this superpoint
            sp_points = pcds[idx]  # shape: (n_points_sp, 3)
            # Farthest point sampling
            chosen_idx_local = farthest_point_sampling(sp_points, sample_k)  # local indices in [0..n_points_sp-1]
            rep_idx = idx[chosen_idx_local]

        else:
            rep_idx = idx
            # Pad if needed so each SP always has sample_k reps
            if n_points_sp < sample_k:
                pad = rep_idx[-1].repeat(sample_k - n_points_sp)
                rep_idx = torch.cat([rep_idx, pad], dim=0)
        reps_list.append(pcds[rep_idx].unsqueeze(0))  # shape: (1, sample_k, 3)
    # Reps: (M, <=sample_k, 3)
    reps = torch.cat(reps_list, dim=0)

    # ================ STEP 3: Compute pairwise min-distances (chunked) ================
    all_dists = torch.empty((M, M), device=device)

    for i in range(M):
        d_row = torch.empty((M,), device=device)
        A = reps[i]  # shape: (Ki, 3), Ki <= sample_k
        for start in range(i, M, batch_size):
            end = min(M, start + batch_size)
            B = reps[start:end]  # shape: (batch_size, Kj, 3)

            d_min_batch = []
            for b_idx in range(B.shape[0]):
                # cdist => shape (Ki, Kj)
                Dij = torch.cdist(A, B[b_idx])
                d_min_batch.append(Dij.min().unsqueeze(0))
            d_min_batch = torch.cat(d_min_batch, dim=0)  # shape: (batch_size,)

            d_row[start:end] = d_min_batch

        d_row[i] = float("inf")  # self-distance
        all_dists[i] = d_row
        # Symmetrize
        if i > 0:
            all_dists[:i, i] = all_dists[i, :i]

    # ================ STEP 4: Radius-based adjacency ================
    adjacency = torch.zeros((M, M), device=device)
    within_radius = (all_dists < radius_threshold)
    adjacency[within_radius] = torch.exp(-all_dists[within_radius])
    adjacency.fill_diagonal_(0.0)

    # ================ STEP 5: Aggregate mask coverage ================
    num_masks_total = sum(m.shape[0] for m in masks_list)
    N = pcds.shape[0]
    mask_coverage = torch.zeros((N, num_masks_total), dtype=torch.bool, device=device)
    mask_confidences = torch.zeros((num_masks_total,), dtype=torch.float, device=device)

    offset = 0
    for masks, mapping, sam_scores in zip(masks_list, mapping_list, sam_scores_list):
        masks = masks.to(device)
        mapping = mapping.to(device)
        num_masks, H, W = masks.shape

        # Points that are "visible" in this frame
        vis = (mapping[:, 3] == 1)
        vis_idx = torch.where(vis)[0]
        y_coords = mapping[vis_idx, 1].long().clamp(0, H - 1)
        x_coords = mapping[vis_idx, 2].long().clamp(0, W - 1)

        point_masks = masks[:, y_coords, x_coords].permute(1, 0)  # (num_vis, num_masks)
        mask_coverage[vis_idx, offset:offset + num_masks] = point_masks

        # Store SAM scores
        sam_scores_tensor = torch.as_tensor(sam_scores, dtype=torch.float, device=device)
        mask_confidences[offset:offset + num_masks] = sam_scores_tensor
        offset += num_masks

    # Group coverage by superpoint (including invalid ones, then filter later)
    _, inverse_indices_all = torch.unique(spt_labels, return_inverse=True)
    sp_counts_all = sp_counts.unsqueeze(1).float()  # (num_superpoints_total, 1)

    # (num_superpoints_total, num_masks_total)
    mask_coverage_grouped = torch.zeros(
        (unique_spts.shape[0], num_masks_total), dtype=torch.long, device=device
    )
    mask_coverage_grouped.index_add_(0, inverse_indices_all, mask_coverage.long())

    # coverage fraction: fraction of each superpoint's points covered by each mask
    mask_fraction = mask_coverage_grouped.float() / (sp_counts_all + 1e-6)

    # ---------------- STEP 5.2: Keep only top_k superpoints per mask ----------------
    #  -- (Optional) If you want to rank by coverage * mask_confidence, you'd do:
    # score_matrix = mask_fraction * mask_confidences.unsqueeze(0)
    # and then sort by score_matrix instead. For now, we just use 'mask_fraction'.

    # Sort each mask (each column) in descending order
    sorted_vals, sorted_indices = torch.sort(mask_fraction, dim=0, descending=True)
    # Create a boolean mask
    topk_mask = torch.zeros_like(mask_fraction, dtype=torch.bool)

    # Mark True for the top_k superpoints in each mask's column
    for m in range(mask_fraction.shape[1]):
        # Indices of top_k rows for mask m
        topk_rows = sorted_indices[:top_k, m]
        topk_mask[topk_rows, m] = True

    # Zero out coverage fraction for superpoints not in top_k for this mask
    mask_fraction = torch.where(topk_mask, mask_fraction, torch.zeros_like(mask_fraction))

    # ================ STEP 6: Build mask_contribution & filter valid superpoints ================
    # Binary: does superpoint have coverage >= mask_thres (but only among top_k now!)
    mask_contribution = (mask_fraction >= mask_thres)

    # Filter to valid superpoints
    sp_masks_valid = mask_contribution[valid_mask]  # (M, num_masks_total)
    sp_masks_raw_valid = mask_fraction[valid_mask]  # (M, num_masks_total)

    # ================ (Optional) Superpoint features ================
    if point_feas is not None:
        point_feas = point_feas.to(device)
        F_dim = point_feas.shape[1]

        sp_features_all = torch.zeros((unique_spts.shape[0], F_dim), device=device)
        sp_features_all.index_add_(0, inverse_indices_all, point_feas)
        sp_features_all /= (sp_counts_all + 1e-6)

        # Normalize for cosine similarity
        sp_features_valid = F.normalize(sp_features_all[valid_mask], p=2, dim=1)
    else:
        sp_features_valid = None

    # ================ STEP 7: Modulate adjacency by mask overlap & optional feature sim ================
    chunk_size = batch_size
    for start_i in range(0, M, chunk_size):
        end_i = min(M, start_i + chunk_size)
        for i in range(start_i, end_i):
            row_vals = adjacency[i]  # shape: (M,)
            j_indices = torch.nonzero(row_vals > 0, as_tuple=True)[0]
            if j_indices.numel() == 0:
                continue

            sp_mask_i = sp_masks_valid[i].unsqueeze(0)  # (1, num_masks_total)
            sp_mask_j = sp_masks_valid[j_indices]  # (num_neighbors, num_masks_total)

            union = (sp_mask_i | sp_mask_j).float()
            n_mask_eff = min(n_mask, union.shape[1])

            # Flatten for topk
            union_flat = union.view(sp_mask_j.size(0), -1)
            _, topk_inds = union_flat.topk(k=n_mask_eff, dim=1)

            sp_mask_i_expanded = sp_mask_i.expand(sp_mask_j.size(0), -1)
            sp_top_masks_i = torch.gather(sp_mask_i_expanded, 1, topk_inds)
            sp_top_masks_j = torch.gather(sp_mask_j, 1, topk_inds)

            inter = (sp_top_masks_i & sp_top_masks_j).sum(dim=1).float()
            uni = (sp_top_masks_i | sp_top_masks_j).sum(dim=1).float()
            overlap_ratios = inter / (uni + 1e-6)

            # Feature similarity (if available)
            if sp_features_valid is not None:
                feat_i = sp_features_valid[i]
                feat_j = sp_features_valid[j_indices]
                dot_prod = torch.sum(feat_i * feat_j, dim=1)
                sim_vals = (dot_prod + 1.0) / 2.0
                sim_vals = torch.clamp(sim_vals, 0.0, 1.0)
                # Where similarity < sim_thres, set factor to 1
                sim_vals = torch.where(sim_vals < sim_thres, torch.ones_like(sim_vals), sim_vals)
            else:
                sim_vals = torch.ones_like(overlap_ratios)

            mod_factor = overlap_ratios * sim_vals

            # adjacency[i, j_indices] = adjacency[i, j_indices] * mod_factor
            adjacency[i, j_indices] = torch.max(mod_factor, adjacency[i, j_indices])
            adjacency[j_indices, i] = adjacency[i, j_indices]  # symmetric

    # ================ STEP 8: Superpoint confidence scores ================
    # Weighted by SAM confidences
    sp_mask_weights = sp_masks_raw_valid * mask_confidences.unsqueeze(0)
    superpoint_mask_confidences = sp_mask_weights.sum(dim=1)
    total_mask_contributions = sp_masks_raw_valid.sum(dim=1) + 1e-6
    superpoint_confidence_scores = superpoint_mask_confidences / total_mask_contributions
    superpoint_confidence_scores = torch.clamp(superpoint_confidence_scores, min=0.0)

    return adjacency, superpoint_confidence_scores



class SAM2Mask(nn.Module):
    def __init__(self, cfg, global_intrinsic=None, class_names=None, device='cuda', data_reader=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        if class_names is not None:
            self.cls_emds = self.text_encoder(class_names)
        else:
            self.cls_emds = None
            self._set_model(cfg.vlm)

    def _set_model(self, config):
        if config.vlm_name == 'llava':
            self.vlm_model, self.vlm_processor = load_llava_model_hg(
                model_path="llava-hf/llava-v1.6-mistral-7b-hf", device=self.device)
            print('llava model loaded')
            self.vlm_model = self.vlm_model.eval()
        elif config.vlm_name == 'idefics':
            model_path = "HuggingFaceM4/idefics2-8b-chatty"
            self.vlm_model, self.vlm_processor = load_idefics_model(model_path, device=self.device)
            self.vlm_model = self.vlm_model.eval()
            print('idefics model loaded')
        else:
            pass

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
        if self.config.vlm.vlm_name == 'llava':
            qs = ['[INST] <image>Can you spot any unusual object in the scene?[/INST]']
            cls_names = get_image_category_from_llavahg([img], self.vlm_model, self.vlm_processor,
                                                        qs, self.device)[0]
        elif self.config.vlm.vlm_name == 'idefics':
            cls_names = get_image_category_from_idefics([img, ], self.vlm_model, self.vlm_processor, device=self.device)
        else:
            raise NotImplementedError
        return cls_names

    def crop_and_extract_features(self, image, detections, thresh=32):
        cropped_tensors = []
        mask_list = []
        scores = []
        labels = []
        for idx, detection in enumerate(detections):
            label = detection.label
            # box = detection.box
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
            labels.append(label)
        masks = torch.stack(mask_list, dim=0)
        scores = torch.tensor(scores).to(self.device)
        # masks_fitted = torch.zeros_like(masks, dtype=bool)
        # masks = torch.logical_and(masks, masks_fitted)  # fitting
        imgs = torch.stack(cropped_tensors).to(self.device)
        img_batches = torch.split(imgs, 64, dim=0)
        image_features = []
        # Batch forwarding CLIP
        with torch.no_grad(), torch.cuda.amp.autocast():
            for img_batch in img_batches:
                image_feat = self.clip_adapter.encode_image(img_batch)
                image_feat /= image_feat.norm(dim=-1, keepdim=True)
                image_features.append(image_feat)
        image_features = torch.cat(image_features, dim=0)

        return image_features, masks, scores

    def text_encoder(self, class_names):
        if class_names is None:
            return None
        with torch.no_grad(), torch.amp.autocast('cuda'):
            text_features = self.clip_adapter.encode_text(
                clip.tokenize(["a " + label for label in class_names]).to(self.device))
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.half()


    def forward(self, scene_id, split='val', class_names=None, bs=256, k_img = 80):
        # Load point cloud data
        points, spp, ins_gt, sem_gt = self.data_reader.get_pcd(scene_id, split)
        # sem_gt[sem_gt<0] = 200
        img_dirs = self.data_reader.get_image_dirs(scene_id, split)
        n_points = points.shape[0]

        spp = num_to_natural_torch(spp, void_number=0)
        visibility = torch.zeros((n_points), dtype=torch.int, device=self.device)
        sem_features = torch.zeros((n_points, self.clip_dim), device=self.device)
        counter = torch.zeros((n_points, 1), device=self.device)

        masks_list = []
        mapping_list = []
        score_list = []

        # If there are more than 120 images, downsample by taking every 2nd image
        if len(img_dirs) > k_img:
            img_dirs = random.sample(img_dirs, k_img)
        num_img = len(img_dirs)
        flag =False
        if class_names is None:
            flag = True
        # Process images
        for img_id, img_dir in enumerate(img_dirs):
            if img_id % 100 == 0:
                print(f'Processing image {img_id}/{num_img}')
            # Load depth and convert to float32 to reduce memory
            depth = self.data_reader.get_depth(img_dir).astype(np.float32) / self.depth_scale
            if (depth.shape[1], depth.shape[0]) != self.img_dim:
                depth = cv2.resize(depth, self.img_dim, interpolation=cv2.INTER_LINEAR)

            # Get pose and intrinsics
            pose = self.data_reader.get_pose(img_dir).astype(np.float32)
            translated_intrinsics = self.data_reader.get_intrinsic(scene_id)
            if flag:
                image = Image.open(img_dir).convert("RGB").resize(self.img_dim)
                cls_names = self.get_image_level_names(image)
                n_objects = len(cls_names)
                if n_objects < 1:
                    print('no cls!!!!!!!!!!!!!!!!')
                    continue
                if n_objects > 0:
                    segment_size = 5  # scannetpp_benchmark
                    segments = [cls_names[i: min(i + segment_size, n_objects - 1)]
                                for i in range(0, len(cls_names), segment_size)]
                else:
                    segments = cls_name
                class_names += cls_names
                detections = []
                for cls_name in segments:
                    detection = self.detect(image, cls_name)
                    boxes = get_boxes(detection)
                    if boxes == [[]]:
                        continue
                    detection = self.segment(image, detection, True)
                    detections.extend(detection)
                try:
                    clip_features, masks, scores = self.crop_and_extract_features(image, detections)
                    visibility[mapping[:, 3] == 1] += 1
                except Exception as e:
                    continue
            else:
                # Get grounded data
                grounded_dict = self.data_reader.get_grounded_data(img_dir, split, scene_id)
                clip_features = grounded_dict["img_feat"]
                masks = grounded_dict["masks"]
                scores = grounded_dict["conf"]

            # If no masks, skip
            if masks is None or len(masks) == 0:
                del depth, pose, translated_intrinsics, grounded_dict, clip_features, masks, scores
                torch.cuda.empty_cache()
                continue

            # Compute mapping (on CPU if memory is too large on GPU)
            mapping = torch.ones([n_points, 4], dtype=int, device=self.device)
            mapping[:, 1:4] = self.pointcloud_mapper.compute_mapping_torch(
                pose, points, depth, intrinsic=translated_intrinsics
            )

            if mapping[:, 3].sum() == 0:
                # No visible points
                del depth, pose, translated_intrinsics, grounded_dict, clip_features, masks, scores, mapping
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

            scores_t = scores.to(self.device)

            masks_list.append(masks_t)
            mapping_list.append(mapping)
            score_list.append(scores_t.view(-1))

            idx = torch.where(mapping[:, 3] == 1)[0]
            visibility[idx] += 1

            # Compute final features
            clip_features = clip_features.to(self.device)
            pred_masks = masks_t.to(self.device)

            # Use half precision if supported to reduce memory
            final_feat = torch.einsum("qc,qhw->chw", clip_features.half(), pred_masks.half())

            # Accumulate features
            sem_features[idx] += final_feat[:, mapping[idx, 1], mapping[idx, 2]].permute(1, 0)
            counter[mapping[:, 3]!=0]+= 1

            # Clean up large variables
            del depth, pose, translated_intrinsics, grounded_dict, clip_features, masks, scores, pred_masks, final_feat, mapping
            torch.cuda.empty_cache()

        # Normalize sem_features
        counter[counter==0] = 1e-5
        sem_features = sem_features / counter
        sem_features = F.normalize(sem_features, dim=-1)

        # Construct adjacency and merge
        mask_tres = 0.3
        n_mask = 8
        sim_thres = 0.8

        adjacency_matrix, superpoint_confidence_scores = construct_superpoint_adjacency_graph_radius(
            points, spp, masks_list, mapping_list, score_list,
            sample_k=72,  # number of representative points to sample per superpoint
            min_points=0,  # minimum number of points required for a valid superpoint
            radius_threshold=0.2,  # number of nearest neighbors to search over the representative points
            point_feas=sem_features,
            mask_thres=mask_tres,
            sim_thres = sim_thres,
            n_mask = n_mask,
            batch_size=32
        )
        # Free lists after use
        del masks_list, mapping_list, score_list, counter
        torch.cuda.empty_cache()
        try:
            n_clusters = 128
            fine_labels, merged_labels, pred_conf = perform_clustering_and_merge_clusters(
                adjacency_matrix, spp, superpoint_confidence_scores,
                n_clusters=n_clusters, k=n_clusters, thres=0.1, tau=0.1, is_auto=False)
        except Exception as e:
            merged_labels, pred_conf = merge_superpoints_based_on_adj_matrix(
                adjacency_matrix, spp, superpoint_confidence_scores, thres=1e-3, is_agg=False
            )
            print(e)
        # Map non-consecutive IDs in merged_labels to consecutive IDs
        unique_labels, new_ids = torch.unique(merged_labels, return_inverse=True)
        num_ins = unique_labels.size(0)

        # Create one-hot encoding
        instance = F.one_hot(new_ids, num_classes=num_ins).T.half()
        # print(instance.shape)

        self.cls_emds = self.text_encoder(class_names)
        if self.cls_emds is not None:
            pred_class_score = torch.zeros((sem_features.shape[0], self.cls_emds.shape[0])).to(sem_features)
            for batch in range(0, sem_features.shape[0], bs):
                start = batch
                end = min(start + bs, sem_features.shape[0])
                pred_class_score[start:end] = (sem_features[start:end].half() @  self.cls_emds.T).softmax(dim=-1)
            ins_class_scores = torch.einsum(
                "kn,nc->kc", instance, pred_class_score.half())  # K x classes
            ins_class_scores = ins_class_scores / instance.sum(dim=1)[:, None]  # K x classes
            sem_pred = torch.argmax(pred_class_score, dim = -1).view(-1)
            # print(torch.unique(sem_pred))
            # print(sem_pred)
            scores_final = ins_class_scores[:, sem_pred]
        else:
            sem_pred = None
            scores_final = pred_conf
        masks_final = instance
        # Cleanup and return
        del adjacency_matrix, superpoint_confidence_scores, spp, points, sem_features
        torch.cuda.empty_cache()
        # print(masks_final.shape, scores_final.shape, ins_gt.shape, sem_gt.shape)
        # AP, AP50, AP25 = compute_ap_metrics(masks_final, scores_final, ins_gt, sem_gt, sem_gt)
        # print(f"AP: {AP:.4f}, AP50: {AP50:.4f}, AP25: {AP25:.4f}")

        return masks_final, scores_final.half(), sem_pred, ins_gt.half(), sem_gt.half()

