import os
import argparse
import torch
import onnx
from models.pfld import PFLDInference


def export_to_onnx(model_path, output_path):
    """Export PyTorch model to ONNX format."""
    # Load PyTorch Model
    print("=====> Loading PyTorch checkpoint...")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return False

    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
    model = PFLDInference()
    model.load_state_dict(checkpoint["pfld_backbone"])
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully.")

    # Convert to ONNX
    print("=====> Converting PyTorch model to ONNX...")
    dummy_input = torch.randn(1, 3, 112, 112)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=True,
        input_names=["input"],
        output_names=["features", "landmarks"],
        opset_version=11,
        do_constant_folding=True
    )

    print(f"ONNX model saved at: {output_path}")

    # Validate ONNX Model
    print("=====> Validating ONNX model...")
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")
        return True
    except Exception as e:
        print(f"ONNX validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Export PFLD model to ONNX format')
    parser.add_argument('--model_path', default="./weights/best_ckpt.pth", type=str,
                       help="Path to the trained PFLD model")
    parser.add_argument('--output_path', default="./weights/pfld_landmark.onnx", type=str,
                       help="Path to save the ONNX model")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    success = export_to_onnx(args.model_path, args.output_path)
    if success:
        print("Export completed successfully!")
    else:
        print("Export failed!")


if __name__ == "__main__":
    main()
