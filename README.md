# TriTAC: Task-Aware Hyperspectral Image Compression with Tri-Domain Latent Modulation

> **Submission:** The manuscript entitled **“TriTAC: Task-Aware Hyperspectral Image Compression with Tri-Domain Latent Modulation”** was submitted to **IEEE Transactions on Geoscience and Remote Sensing (TGRS)** on **October 23, 2026**.

This repository provides the core training, evaluation, and entropy coding workflows for TriTAC.

The release supports HySpecNet-11k reconstruction and Indian Pines classification. The released configuration uses 100 input bands, a latent channel multiplier of 4, and spectral, spatial, and frequency TriLM branches at both the encoder and decoder. Training and deployment share the model implementation in `lvae/`; deployment entry points are in `deployment/`.

## Downloads

Download the example datasets and model checkpoints from Google Drive:

| Resource | Download | Local destination |
| --- | --- | --- |
| Example datasets | [Download datasets](https://drive.google.com/drive/folders/1yy9Ys4s2aEqHieFDeP0_NUivuUcm74Qz?usp=drive_link) | `data/` |
| Model checkpoints | [Download weights](https://drive.google.com/drive/folders/1_TX0SAynaWaxB1nQ7XVUvTgnc1x88dtF?usp=drive_link) | `weights/` |

Extract downloaded archives, if applicable, and arrange the files as shown below. Avoid extra nested folders such as `data/data/` or `weights/weights/`. The default commands expect these paths relative to the repository root:

```text
TriTAC_public/
  data/
    hyspecnet11k/
      manifest.json
      test/
        hyspecnet11k_test_*.npy
    indian_pines/
      Indian_pines_corrected.mat
      Indian_pines_gt.mat
      Indian_pines_uint8.npy
      Indian_pines_gt.npy
  weights/
    stage1_hyspec_l4096.pt
    stage1_indian_l4096.pt
    stage2_indian_l4096.pt
    hypersigma_finetuned_indian.pt
    indian_pines_classification_head.pt
    manifest.json
```

The dataset release contains **112 HySpecNet-11k test examples and the complete Indian Pines scene**. The weight release contains four complete model checkpoints and one separately exported classification head. The full HySpecNet training set and the original HyperSIGMA initialization checkpoints are not included.

## 1. Environment

Validated locally with Python 3.10, PyTorch 2.5.1 + CUDA 12.1, torchvision 0.20.1, and an RTX 3080 with 10 GB of memory. Run all commands from the repository root.

```bash
conda create -n tritac python=3.10 -y
conda activate tritac
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

On Orin, install PyTorch and torchvision versions compatible with your JetPack installation instead of the desktop CUDA wheels above. Entropy coding requires working CompressAI C++ extensions. Validation of this release was performed on the desktop GPU; Orin hardware measurements were not repeated.

## 2. Code and datasets

| Entry point | Purpose |
| --- | --- |
| `train_stage1.py` | Train the compression backbone |
| `train_stage1_adapt.py` | Adapt the codec to Indian Pines without labels |
| `train_task.py` | Fine-tune the HyperSIGMA classification model |
| `train_stage2.py` | Train TriLM while freezing both backbones |
| `test_stage1.py` | Evaluate HySpecNet reconstruction |
| `test_stage2.py` | Evaluate Indian Pines classification and reconstruction |
| `deployment/codec.py` | Encode and decode actual bitstreams |
| `deployment/benchmark.py` | Measure bitstream size and codec latency |

Supporting code is in `lvae/` and `third_party/HyperSIGMA/`. Ablations, parameter sweeps, detection experiments, and historical experiment logs are excluded from this release.

### HySpecNet-11k examples

A reproducible random subset of `ceil(1118 * 0.1) = 112` files was selected from 1,118 test files using seed 3704. Selected filenames are recorded in `data/hyspecnet11k/manifest.json`. Each original array is stored as uint8 with shape 202 × 128 × 128. The model uses the first 100 bands, normalized by 255.

These examples are for testing the released pipeline. They are not a training set or a replacement for the full test set used in the paper.

### Indian Pines

The complete scene and labels are provided in both the original supplied `.mat` files and the `.npy` representation used by the experiments. Default scripts load `Indian_pines_uint8.npy` (145 × 145 × 200) and `Indian_pines_gt.npy`, selecting the first 100 bands and dividing by 255. The `.mat` files are provided as data attachments and are not loaded directly by these scripts.

The classification split uses seed 3704, with 10 training and 5 validation pixels per class; the remaining labeled pixels form the test set. The resulting counts are 160 / 80 / 10,009. Label 0 is excluded from evaluation. Scene adaptation and PCA use the full unlabeled image, supervised training uses training labels only, and model selection uses validation labels. This is a single-scene transductive setting, not a cross-scene generalization evaluation.

## 3. Released checkpoints

Download the checkpoints using the **Download weights** link above and place them in `weights/`.

| File | Purpose | Approximate size |
| --- | --- | --- |
| `stage1_hyspec_l4096.pt` | Compression backbone pretrained on HySpecNet | 381 MiB |
| `stage1_indian_l4096.pt` | Compression backbone adapted to Indian Pines | 381 MiB |
| `stage2_indian_l4096.pt` | Complete codec with task-adapted TriLM | 393 MiB |
| `hypersigma_finetuned_indian.pt` | Complete fine-tuned HyperSIGMA, including the classifier, PCA transform, and split configuration | 753 MiB |
| `indian_pines_classification_head.pt` | Final classifier mapping 128-dimensional features to 16 classes | 10 KiB |

This release includes one L4096 configuration. Model tensors are unchanged from the source checkpoints; optimizer states and personal paths were removed. Stage-2 evaluation does not require the original HyperSIGMA initialization checkpoints. Provenance, selected steps, and SHA-256 checksums are recorded in `weights/manifest.json`.

The standalone classification head is an exact export of `model.cls` from `hypersigma_finetuned_indian.pt`. It has 2,064 parameters: the weights and bias of a 1 × 1 convolution. It was fine-tuned during task-model preparation and remained frozen during stage 2. The spatial/spectral backbones and fusion modules remain in the complete model. Standard evaluation already loads this classifier as part of the complete checkpoint.

To load the classifier separately:

```python
from pathlib import Path
import torch
from tritac import load_task

task, projector, cfg, seed = load_task(
    Path("weights/hypersigma_finetuned_indian.pt"), torch.device("cpu")
)
head = torch.load(
    "weights/indian_pines_classification_head.pt",
    map_location="cpu", weights_only=True
)
task.model.cls.load_state_dict(head["model"], strict=True)
```

**Configuration note:** L4096 refers to the stage-1 rate-distortion coefficient. The released stage-2 checkpoint records `task_lambda=1.0` and `recon_lambda=0.01`; the training defaults preserve this configuration. Its objective is:

```text
rate_loss + task_lambda * feature_MSE + recon_lambda * reconstruction_MSE
```

The stage-1 distortion term uses the model's internal [-1, 1] scale; the stage-2 reconstruction MSE uses [0, 1]. The 4096 in the filename is not the stage-2 reconstruction-loss coefficient.

## 4. Evaluation

After downloading and arranging the data and weights, test one example, then all 112 examples and the Indian Pines scene:

```bash
python test_stage1.py --max-files 1
python test_stage1.py
python test_stage2.py
```

Results are written to `outputs/stage1.json` and `outputs/stage2.json`. To evaluate classification with the stage-1 scene-adapted codec:

```bash
python test_stage2.py --checkpoint weights/stage1_indian_l4096.pt --output outputs/stage1_indian.json
```

These scripts report estimated entropy rates. Use Section 6 for actual bitstream sizes. PSNR uses a peak value of 1, SAM is reported in degrees, and classification OA is stored on a 0–1 scale.

Local checks with the released files produced:

| Evaluation | Result |
| --- | --- |
| Mean PSNR on the 112 HySpecNet examples | 37.2933 dB |
| Indian Pines stage-2 reconstruction PSNR | 26.4989 dB |
| Indian Pines classification OA after reconstruction | 80.7473% |
| Indian Pines classification OA on the original image | 81.1570% |

Validation also covered one training step each for stage 1, scene adaptation, and stage 2; one epoch of task-model training; entropy coding at depths 8 and 9; and decoding in a separate process. Full training was not repeated during release preparation.

## 5. Training

### Stage 1: Compression backbone

Prepare a separate directory of HySpecNet training `.npy` files and replace `YOUR_TRAIN_NPY_DIR` below. The script creates a reproducible training/validation split. Do not use the distributed test examples for formal training.

```bash
python train_stage1.py --data YOUR_TRAIN_NPY_DIR --steps 10000 --batch-size 32 --crop-size 128 --latent-scale 4 --rd-lambda 4096 --save-dir runs/stage1
```

Reduce `--batch-size` if GPU memory is insufficient. The best checkpoint is selected by validation PSNR and saved to `runs/stage1/base/latent4x/lambda4096/best.pt`.

### Stage 1: Target-scene adaptation

```bash
python train_stage1_adapt.py --checkpoint weights/stage1_hyspec_l4096.pt --dataset indian --steps 2000 --batch-size 4 --crop-size 128 --samples-per-epoch 64 --val-interval 100 --val-steps 4 --save-dir runs/domain_adapt
```

This step performs rate-distortion fine-tuning using the image without classification labels. The output is `runs/domain_adapt/indian/lambda4096/best.pt`. If you trained your own compression backbone, pass its checkpoint instead.

### Task-model preparation

Skip this step when using the released fine-tuned HyperSIGMA checkpoint. Otherwise, obtain the official spatial and spectral base initialization checkpoints from the HyperSIGMA project and replace the two paths below:

```bash
python train_task.py --hypersigma-spat-ckpt SPAT_PRETRAINED.pth --hypersigma-spec-ckpt SPEC_PRETRAINED.pth --output runs/task/hypersigma_finetuned_indian.pt
```

The output contains the task model, PCA transform, and split seed. Training defaults to 500 epochs, with checkpoint selection based on validation loss. Test labels are not used for model selection.

### Stage 2: Task-aware adaptation

```bash
python train_stage2.py --steps 5000 --val-interval 500 --batch-size 1 --save-dir runs/stage2
python test_stage2.py --checkpoint runs/stage2/best.pt --output outputs/retrained_stage2.json
```

By default, this loads the released scene-adapted codec and task model. All three TriLM branches at both ends are trained, totaling approximately 3.08 M parameters, while both backbones remain frozen. The best checkpoint is selected by validation OA; the selected model is evaluated on the test split after training.

To use your own outputs from the preceding steps:

```bash
python train_stage2.py --stage1-checkpoint runs/domain_adapt/indian/lambda4096/best.pt --task-checkpoint runs/task/hypersigma_finetuned_indian.pt --save-dir runs/stage2
```

When evaluating this model, also pass your corresponding task checkpoint:

```bash
python test_stage2.py --checkpoint runs/stage2/best.pt --task-checkpoint runs/task/hypersigma_finetuned_indian.pt --output outputs/retrained_stage2.json
```

## 6. Deployment: Actual bitstreams

With fixed model weights, `--depth` selects the first d latent levels, from 1 to 9. Omitted levels are replaced by their conditional prior means. No retraining is needed for each depth; this operation is not arbitrary truncation of an already encoded file.

```bash
python -m deployment.codec encode data/indian_pines/Indian_pines_uint8.npy outputs/indian_d8.tritac --depth 8
python -m deployment.codec decode outputs/indian_d8.tritac outputs/indian_d8.npy
python -m deployment.benchmark --depth 8 --repeat 5 --warmup 1 --classification --output outputs/benchmark_d8.json
```

The encoder and decoder must use the same weights and compatible software environments. Decoding returns a float32 CHW array with 100 bands in [0, 1]. The default tile size is 128 × 128. Edge tiles are padded to multiples of 64 and cropped back after decoding. The container stores shape information, tile metadata, and native entropy-coded streams; it does not contain model weights.

Benchmark fields:

- `native_bpppb`: total native tile-stream bytes, including their required headers, multiplied by 8 and divided by the original number of band-pixels.
- `container_bpppb`: actual `.tritac` file bytes multiplied by 8 and divided by the original number of band-pixels, including ZIP_STORED and JSON overhead.
- `median_encode_decode_ms`: median encoding-plus-decoding time after warmup, with CUDA synchronization. Includes tensor transfers and in-memory packaging; excludes model loading, entropy-table initialization, disk I/O, and classification.

Entropy-coding failures raise errors rather than falling back to estimated rates. Latencies measured with different hardware, tile sizes, or timing boundaries should not be directly compared with the paper's Orin measurements.

## 7. Acknowledgments and file distribution

This implementation builds on [QARV / LoSSy-VAE](https://github.com/duanzhiihao/lossy-vae) and [HyperSIGMA](https://github.com/WHU-Sigma/HyperSIGMA). See `third_party/README.md` and `third_party/HyperSIGMA/LICENSE` for attribution and the retained license. The QARV upstream README specifies non-commercial use; this release does not relicense third-party components under MIT or Apache.

Datasets and checkpoints can be downloaded separately using the Google Drive links at the top of this README. Preserve the documented local directory structure. Git LFS patterns are also provided in `.gitattributes` if binary files are tracked in Git; initialize Git LFS before adding them. Generated `runs/`, `outputs/`, and Python caches are excluded by `.gitignore`.
