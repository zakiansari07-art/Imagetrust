import torch
from dataloader import create_dataloader

train_dataloader , validation_dataloader = create_dataloader()

image, label = next(iter(train_dataloader))

print(f"Image Shape: {image.shape}, Label Shape: {label.shape}")

print(f"Image Type: {image.dtype}, Label Shape: {label.dtype}")

print()