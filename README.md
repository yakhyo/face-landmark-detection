# PFLD: Practical Facial Landmark Detector

A PyTorch implementation of PFLD (Practical Facial Landmark Detector) for real-time facial landmark detection.

## Features

- Real-time facial landmark detection with 98 landmarks
- Lightweight MobileNet-based architecture
- Training and evaluation on WFLW dataset
- Webcam inference support
- ONNX model export
- Comprehensive evaluation metrics (NME, AUC, failure rate)


## Quick Start

### Training

Train the model on WFLW dataset:

```bash
python train.py --train-data ./data/train_data/list.txt \
                --val-data ./data/test_data/list.txt \
                --batch-size 256 \
                --num-epochs 500 \
                --base-lr 1e-4
```

### Testing

Evaluate the trained model:

```bash
python test.py --model-path ./weights/best_ckpt.pth \
               --test-dataset ./data/test_data/list.txt \
               --show-image
```

### Inference

Run inference on webcam:

```bash
python inference.py --model_path ./weights/best_ckpt.pth
```

Run inference on a single image:

```bash
python inference.py --model_path ./weights/best_ckpt.pth --image path/to/image.jpg
```

Save result to file:

```bash
python inference.py --model_path ./weights/best_ckpt.pth --image path/to/image.jpg --output result.jpg
```

### ONNX Export

Export model to ONNX format:

```bash
python onnx_export.py --model_path ./weights/best_ckpt.pth --output_path ./weights/pfld_landmark.onnx
```

### ONNX Inference

Run ONNX inference on webcam:

```bash
python onnx_inference.py --model_path ./weights/pfld_landmark.onnx
```

Run ONNX inference on a single image:

```bash
python onnx_inference.py --model_path ./weights/pfld_landmark.onnx --image path/to/image.jpg
```

## Dataset Format

The dataset list file should contain one line per sample:
```
image_path landmark_x1 landmark_y1 ... landmark_x98 landmark_y98 attr1 attr2 ... attr6 euler_x euler_y euler_z
```

Where:
- `image_path`: Path to the image file
- `landmark_xi landmark_yi`: 98 facial landmark coordinates (196 values total)
- `attr1-attr6`: Facial attributes (6 values)
- `euler_x euler_y euler_z`: Head pose angles (3 values)

## Model Architecture

The PFLD model consists of:
- **Backbone**: MobileNet-based feature extractor with inverted residual blocks
- **Multi-scale Features**: Features from different stages are combined
- **Auxiliary Network**: Additional network for pose estimation during training

## Training Configuration

Key training parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch-size` | 256 | Training batch size |
| `--base-lr` | 1e-4 | Initial learning rate |
| `--weight-decay` | 1e-6 | Weight decay for optimizer |
| `--num-epochs` | 500 | Total training epochs |
| `--lr-patience` | 40 | LR scheduler patience |

## Evaluation Metrics

The model is evaluated using:
- **NME (Normalized Mean Error)**: Average landmark error normalized by inter-ocular distance
- **AUC**: Area under the cumulative error distribution curve
- **Failure Rate**: Percentage of samples with NME > 0.1
