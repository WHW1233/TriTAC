# TriTAC: Tri-domain Modulation for Task-aware Compression

This repository contains the official implementation for **TriTAC**, a novel hyperspectral image compression framework. TriTAC integrates **SpMA (Spectral Modulation Adapter)**, **SMA (Spatial Modulation Adapter)**, and **FMA (Frequency Modulation Adapter)** to progressively refine latent representations for both reconstruction and downstream analytical tasks.

## Features

- **Tri-domain Modulation**: Effectively captures spectral, spatial, and frequency characteristics inherent to hyperspectral imaging (HSI).
- **Task-aware Compression**: Simultaneously optimizes for bit-rate, reconstruction quality (MSE, SAM), and downstream task performance.
- **Dynamic Adaptability**: Easy integration of classification, segmentation, or other analytical tasks with joint loss functions.

## Dataset

We use the [HySpecNet-11k](https://www.rsim.tu-berlin.de/menue/research/datasets/hyspecnet_11k/) dataset, a large-scale hyperspectral dataset.

## Training

To train the TriTAC framework on CPU/GPU, use the integrated `train_dcae_hsi_task.py` script:

```bash
python train_dcae_hsi_task.py \
    -d /path/to/hyspecnet11k \
    --test_dataset /path/to/hyspecnet11k/test_data \
    --in_channels 202 \
    --eval_freq 1 \
    --batch-size 8 \
    --epochs 50 \
    --task_type classification
```

### Key Arguments
- `--in_channels`: Number of spectral bands in the hyperspectral imagery (e.g., 202).
- `--task_type`: Downstream task to optimize (e.g., `classification` or `segmentation`).
- `--lambda`: Bit-rate distortion parameter for controlling the compression ratio.
- `--lambda_m`: Task loss weight for joint optimization.
- `--alpha`: Trade-off weight between SAM and MSE in human-vision distortion metric.

## Evaluation Metrics

During training, the framework automatically evaluates and logs:
- **Bpppb**: Bits per pixel per band.
- **PSNR / MSE**: Pixel-wise reconstruction fidelity.
- **SAM**: Spectral Angle Mapper for spectral signature fidelity.
- **Task Loss**: Task-specific loss depending on the adapter.

## License

This project is licensed under the MIT License.
