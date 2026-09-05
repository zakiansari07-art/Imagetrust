import torch
import torchvision.models as models


print("=" * 50)
print("ImageTrust - GPU Test")
print("=" * 50)

# Check PyTorch
print(f"PyTorch version: {torch.__version__}")

# Check CUDA
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if not cuda_available:
    print("\nWARNING: CUDA is not available!")
    print("The model will run on CPU.")
    device = torch.device("cpu")
else:
    device = torch.device("cuda")

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    gpu_memory = (
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3
    )

    print(f"GPU memory: {gpu_memory:.2f} GB")


print("\nLoading pretrained ResNet-18...")

# Load a pretrained ResNet-18
model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Move model to GPU
model = model.to(device)

print("Model loaded!")
print(f"Model device: {next(model.parameters()).device}")


# Create a fake image
# Shape: batch, channels, height, width
image = torch.randn(
    1,
    3,
    224,
    224,
    device=device
)

print(f"Input device: {image.device}")
print(f"Input shape: {image.shape}")


# Run inference
with torch.no_grad():
    output = model(image)


print(f"Output shape: {output.shape}")

print("\n" + "=" * 50)
print("GPU TEST COMPLETE")
print("=" * 50)