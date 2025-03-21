import os
import torch
import onnx
from models.pfld import PFLDInference

# Define paths (Update if needed)
torch_model_path = "./checkpoint/last_ckpt.pth"
onnx_model_path = "./checkpoint/pfld_landmark.onnx"

# Load PyTorch Model
print("=====> Loading PyTorch checkpoint...")
checkpoint = torch.load(torch_model_path, weights_only=False, map_location="cpu")
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
    onnx_model_path,
    verbose=True,
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    do_constant_folding=True
)

print(f"ONNX model saved at: {onnx_model_path}")

# Validate ONNX Model
print("=====> Validating ONNX model...")
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")
