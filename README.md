# Dance Style Recognition

Unofficial implementation of the paper *Dance Style Recognition Using Laban Movement Analysis* by Turab et al.:
https://arxiv.org/abs/2504.21166

This repository extracts per-frame LMA (Laban Movement Analysis) features from
dance videos and trains a GPU-accelerated classifier to recognize dance styles
on the AIST++ dataset. The extractor emits **61 features** — the 55 of Turab et
al. plus six inter-joint angles (`Angle_{LArm,RArm,Shoulders,LKnee,RKnee,Hips}`).
Per-joint dynamics use a lag-1 (`derivative="gradient"`) finite difference by
default.

> This is also the LMA extractor used by *Appearance-Invariant Detection of
> Suggestive Motion via Laban Movement Descriptors* (SIGGRAPH Posters '26,
> [doi:10.1145/3799825.3818709](https://doi.org/10.1145/3799825.3818709)); see
> [zaiisao/suggestive-motion-lma](https://github.com/zaiisao/suggestive-motion-lma).

## Repository Layout
- `src/process_lma_features.py`: LMA feature extraction from videos.
- `src/train_lma.py`: GPU-accelerated training and evaluation.
- `models/`: local model files required for feature extraction.
- `environment.yml`: conda environment definition.

## Setup
1) Create the conda environment:
```bash
conda env create -f environment.yml
conda activate dance-recognition
```

2) Ensure model checkpoints are available in the models/ directory:
- `nlf_l_multi_0.3.2.torchscript`
- MoGe model is pulled via Hugging Face in code, but may require login
	depending on your local configuration.

## Feature Extraction
Extract LMA features from a single video:
```bash
python src/process_lma_features.py \
	--input_path /path/to/video.mp4 \
	--output_dir /path/to/output
```

Extract LMA features from a folder of videos:
```bash
python src/process_lma_features.py \
	--input_path /path/to/video_folder \
	--output_dir /path/to/output
```

Outputs:
- `<video>_features.npy`: per-window LMA matrix of shape `(num_windows, 61)`
- `<video>_pose.npz`: cached raw joints + per-clip floor model, so any feature-set
  variant can be recomputed on CPU without re-running the pose/depth stack

## Training and Evaluation
Train with video-level splits (`GroupKFold`, recommended):
```bash
python src/train_lma.py \
	--data_dir /path/to/output \
	--mode original
```

Train with shuffled frame-level splits (optimistic baseline):
```bash
python src/train_lma.py \
	--data_dir /path/to/output \
	--mode shuffled
```

Save best model per fold:
```bash
python src/train_lma.py \
	--data_dir /path/to/output \
	--mode original \
	--save_models /path/to/save_models
```

## Notes
- The training script expects filenames with AIST++ genre codes such as gBR,
	gHO, gJB, etc., and the suffix *_features.npy.
- GPU acceleration uses CuPy and cuML. A compatible CUDA toolkit and driver
	are required.
- This is an unofficial implementation and may diverge from the paper.

