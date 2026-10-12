# PoVo

Official implementation of **Vocabulary-Free 3D Instance Segmentation with Vision and Language Assistant**, accepted by **3DV 2025**.

[Project website](https://gfmei.github.io/PoVo/) · [Paper](https://www.arxiv.org/pdf/2408.10652) · [GitHub](https://github.com/gfmei/PoVo)

This repository provides a configuration using **50 RGB-D views per scene**: **SAM2.1 → point-curve graph → constrained spectral clustering → ViT-L instance crops → Qwen3.5 reranking**. ScanNet200 preparation, evaluation, regression tests, and visualization tools support this workflow. Use **all available views** when aiming for better results.

The original **LLaVA → Grounding DINO → SAM → point-curve graph → constrained spectral clustering** vocabulary-free pipeline is also available with a 50-view configuration.

## Pipeline

1. Select **50 camera-diverse RGB-D views** per scene and generate masks with **SAM2.1 Hiera large**.
2. Fuse **CLIP ViT-B/32** mask features into points. Morton and Hilbert windows propose superpoint neighbors from their actual points; minimum boundary distances verify each edge.
3. Form instances with quality-based spectral selection, `kmeans` using four deterministic Lloyd starts, frame-mask cannot-link constraints, and constrained average-linkage merging.
4. Encode native-resolution crops from up to three diverse views at three context scales with matched **CLIP ViT-L/14@336px** image/text encoders. Give equal weight to context crops and crops focused on the instance, and average six normalized text prompts per class.
5. Use **Qwen3.5-9B** to rerank ambiguous instances among their top-five supplied labels. Score the selected label with its CLIP probability multiplied by instance objectness. Invalid responses retain the CLIP label.

This is **provided-vocabulary open-vocabulary instance segmentation**: every scan receives the same **198 ScanNet200 foreground names**. Qwen recognizes candidates rather than discovering categories. Ground truth is used only for evaluation. Instances without usable crops retain aligned point-feature evidence.

## Installation

Use Python 3.11 and a CUDA-compatible PyTorch environment.

```bash
git clone https://github.com/gfmei/PoVo.git
cd PoVo
python -m pip install -r requirements.txt -r requirements-vlm.txt
```

Place SAM2.1 Hiera large weights at `checkpoints/sam2.1_hiera_large.pt`. CLIP downloads its weights on first use. Qwen uses NF4 double quantization and checkpoint revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, as specified in [configs/scannet200.yaml](configs/scannet200.yaml). One loaded Qwen model handles all scenes in a recognition run.

## Prepare ScanNet200

Raw data must include `scans/SCENE/SCENE.sens`, meshes, segmentation and aggregation annotations, and `scannetv2-labels.combined.tsv`.

```bash
python tools/prepare_cache.py \
  --raw-root /path/to/raw/scannet \
  --output-root /path/to/prepared/cache \
  --full-validation --frames 50
```

This prepares every scan in the official validation list and checks required inputs before resuming. The cache contains:

```text
CACHE/
├── manifest.json
├── SCENE_preparation.json
├── scannet200/val/SCENE.pth
├── scannet_2d/val/SCENE/{color,depth,pose}/
├── intrinsic/SCENE.txt
└── masks/val/SCENE/sam2/FRAME.pth
```

Point files contain `coords`, `spts`, `instance_ids`, and `labels`. Mask caches contain `masks`, `img_feat`, and `conf`. Cached graph features use ViT-B/32 with 512 dimensions; crop recognition uses ViT-L with 768 dimensions.

## Run

```bash
python tools/recognize_instances.py \
  --data-root /path/to/prepared/cache \
  --raw-root /path/to/raw/scannet \
  --output-root /path/to/recognition-results \
  --cohort validation --frames 50 \
  --vlm-cache-dir /path/to/huggingface/cache
```

The command prepares instances, extracts crop embeddings, reranks labels, and evaluates the complete cohort. Its `prepare`, `clip`, `rerank`, and `report` phases also allow individual stages to resume. Source, configuration, image, and mask fingerprints guard completed caches; changed inputs or source require a fresh output root.

Each scene's `predictions.pth` contains point-to-instance assignments, zero-based foreground class IDs (`-1` for unknown), scores, and class names. `validation_summary.json` records semantic and class-agnostic AP, per-scene and per-class metrics, head/common/tail groups, and localized class accuracy. AP is aggregated across the full split.

Target crops preserve foreground colors and retain 25% of the background color while blending it toward gray. A SAM mask supplies the target when it agrees with visible projected points; otherwise projected points supply it. Qwen uses up to two context crops per request, greedy decoding, thinking disabled, and a 64-token limit.

| Setting | Selected value |
| --- | --- |
| Views per scene | 50 |
| Cluster budget / merge threshold | 128 / 0.5 |
| Neighbor radius / curve window | 0.2 / 32 |
| Curve orders | Morton, Hilbert, Morton `yzx`, Hilbert `yzx` |
| Mask coverage / shifted cosine threshold | 0.3 / 0.8 |
| Focused / context crop weight | 0.5 / 0.5 |
| Text prompts per class / reranking candidates | 6 / 5 |

## Graph weighting with label histograms and RGB

The [histogram/RGB configuration](configs/scannet200_histogram_rgb.yaml) adds semantic and color evidence to the superpoint graph:

1. Project each superpoint into the images and rank masks by **IoU with its visible projected pixels**. Keep only the **top `m` masks across all views**, with `graph.mask_top_m: 5` by default. Duplicate pixel projections count once; occluded or insufficiently visible superpoints cast no vote.
2. Compare only the selected masks' CLIP features with text embeddings of the current vocabulary. Accumulate their top-k label probabilities, weighted by **mask overlap × mask confidence**, into a normalized **label histogram per superpoint**. Compare histograms with cosine similarity. `graph.semantic_top_k` controls category candidates per selected mask independently of `m`; set it to `1` for one predicted label per mask.
3. Pool the cached mesh RGB colors per superpoint and compare mean colors with a Gaussian kernel. Colors come from the input scan; annotations are not used.
4. Blend histogram, visual-feature, and RGB similarities with starting weights **0.5 / 0.25 / 0.25**. Missing cues are omitted and weights renormalized for each pair.

The edge weight is **spatial affinity × frame-mask agreement × blended appearance similarity**, with the configured similarity threshold applied to the blend. Spatial neighbors still require verified boundary distances, and repeated mask-separation constraints still remove edges. RGB bandwidth, top-m mask selection, top-k category candidates, and all three modality weights are configurable under `graph`. Mask-agreement and separation evidence still use all valid views; the top-m cap controls the semantic histogram.

Use a fresh prediction directory with the same prepared cache:

```bash
python tools/recognize_instances.py \
  --data-root /path/to/prepared/cache \
  --raw-root /path/to/raw/scannet \
  --output-root /path/to/histogram-rgb-predictions \
  --cohort validation --frames 50 \
  --config configs/scannet200_histogram_rgb.yaml
```

Both the SAM2 and original LLaVA pipelines support `graph.appearance_mode: histogram_rgb`. For LLaVA, histograms use the scene's discovered names; enable the mode and its weights in `configs/scannet200_llava.yaml` before running segmentation.

The histogram/RGB mode is experimental. The measured configurations below use `graph.appearance_mode: clip`.

## 50-view results

The **SAM2.1 + ViT-L instance crops + Qwen3.5 reranking** workflow was evaluated on the complete official **ScanNet200 validation split: 312 scans, 50 RGB-D views per scan, 198 supplied foreground categories, and 16,910 predicted instances**.

| Evaluation | AP (%) | AP50 (%) | AP25 (%) |
| --- | ---: | ---: | ---: |
| Semantic instance segmentation | **12.40** | **19.97** | **26.44** |
| Class-agnostic instance segmentation | **12.21** | **23.91** | **42.09** |

Semantic AP averages the **183 classes with eligible ground truth**. Localized class accuracy is **61.09% (2,291/3,750)** for predicted instances with ground-truth IoU ≥ 0.5. Class-agnostic evaluation uses instance objectness and ignores recognition labels.

These measurements use the 50-view SAM2/Qwen workflow. Settings were selected on validation scenes, which remain included in the complete split. The [verified metrics and protocol](static/results/validation.json) record the measured source revision and report hash.

## Improved instance clustering

The [mask graph configuration](configs/scannet200_mask_graph.yaml) improves class-agnostic instance segmentation on the same **312 validation scans with 50 views each**:

| Configuration | AP (%) | AP50 (%) | AP25 (%) |
| --- | ---: | ---: | ---: |
| Baseline | 12.21 | 23.91 | 42.09 |
| Weighted graph merging | **12.76** | **24.77** | **42.96** |

During merging, spatial neighbor pairs retain their full weight, including neighbors with zero agreement. Non-neighbor pairs contribute `0.05` to the normalization count. Edge sums and effective pair counts accumulate after each merge. Local mask-separation constraints require conflicts in at least half of jointly observed views; repeated global separation constraints remain mandatory.

Use `--config configs/scannet200_mask_graph.yaml` with a **fresh output root** in the recognition command above. The default configuration retains the baseline settings. Both SAM2 and the original LLaVA code support the new merging parameter; this accuracy comparison evaluates SAM2 masks.

Parameters were selected on validation subsets included in the complete split. These are mask-only measurements; semantic recognition and Qwen reranking have not been retested on the changed masks. Recall at IoU 0.25 decreases slightly despite the higher AP. The [paired metrics and verification protocol](static/results/mask_graph_50.json) record the complete comparison.

## Use all views

The default 50-view configuration limits scene coverage. To use all available RGB-D views, prepare a separate cache and pass `--frames all` to both commands:

```bash
python tools/prepare_cache.py \
  --raw-root /path/to/raw/scannet \
  --output-root /path/to/all-views/cache \
  --full-validation --frames all
python tools/recognize_instances.py \
  --data-root /path/to/all-views/cache \
  --raw-root /path/to/raw/scannet \
  --output-root /path/to/all-views/predictions \
  --cohort validation --frames all \
  --vlm-cache-dir /path/to/huggingface/cache
```

All-view selection retains every frame with a finite camera pose. A 50-view cache must be regenerated to include the remaining views. Internally, a frame limit of `0` represents all views.

## Original LLaVA pipeline

[models/povo.py](models/povo.py) restores image-only vocabulary discovery with **LLaVA-1.6-Mistral-7B**, followed by Grounding DINO detection, SAM segmentation, CLIP point-feature fusion, and the shared graph and clustering code. Each image supplies its own object names; no fixed ScanNet class list is given to LLaVA or the detector. The [50-view configuration](configs/scannet200_llava.yaml) pins the LLaVA checkpoint.

Install the LLaVA runtime:

```bash
python -m pip install -r requirements.txt -r requirements-llava.txt
```

Use the prepared RGB-D cache above. Discover and save object names, then run segmentation with those saved names:

```bash
python tools/discover_vocabulary.py \
  --data-root /path/to/prepared/cache \
  --scenes-file datasets/meta_data/scannetv2_val.txt \
  --output-root /path/to/llava-vocabulary \
  --frames 50 --phase discover \
  --cache-dir /path/to/huggingface/cache
python tools/evaluate_vocabulary.py \
  --data-root /path/to/prepared/cache \
  --vocabulary-root /path/to/llava-vocabulary \
  --scenes-file datasets/meta_data/scannetv2_val.txt \
  --config configs/scannet200_llava.yaml \
  --output-root /path/to/llava-predictions
```

LLaVA processes images independently and decodes only generated answer tokens. The parser preserves compound category names and bounds each list to 20 categories. The segmentation command reuses the saved vocabularies, verifies the images and camera selection, and maps predicted names to ScanNet200 categories only for evaluation. Names outside that taxonomy remain unknown.

For all-view LLaVA inference, use the all-view cache and `--frames all` in the discovery command. Segmentation follows the view selection saved with the vocabularies. The model in `models/povo.py` also supports live LLaVA discovery when no saved vocabulary callback is supplied.

## Visualization

The [interactive viewer](https://gfmei.github.io/PoVo/#explore) shows **scene0645_01** as RGB, ground-truth classes, and predictions from the 50-view configuration.

The display uses 30,000 deterministic sampled points. Rotate, zoom, inspect labels, switch between class and instance colors, or clip the room height.

```bash
python tools/export_web_examples.py \
  --data-root /path/to/prepared/cache \
  --prediction-root /path/to/recognition-results \
  --output-root static/examples \
  --scenes scene0645_01
python -m http.server 8000
```

Open `http://localhost:8000`. Full datasets, predictions, caches, reports, logs, and downloaded weights remain outside Git. ScanNet data is subject to its [terms of use](https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf).

## Tests

```bash
python -m pip install -r requirements-test.txt
python -m pytest
```

Tests cover preparation, view selection, projection, fusion, point-curve neighbors, clustering boundaries and fallbacks, matched crop recognition, Qwen and LLaVA generation interfaces, vocabulary parsing and cache validation, evaluation, and visualization alignment. Tests use synthetic inputs and model stubs without downloading weights or rerunning dataset experiments.

## Contact and license

Open an [issue](https://github.com/gfmei/PoVo/issues) or contact [gmei@fbk.eu](mailto:gmei@fbk.eu).

Code is licensed under [Apache 2.0](LICENSE); the project website uses [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Retained third-party notices are in `sam2/LICENSE`, `libs/hilbert_LICENSE.txt`, and `libs/serialization_LICENSE.txt`.
