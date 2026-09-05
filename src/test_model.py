import torch
from data.dataloader import create_dataloader
from models.detector import create_model

#Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

#Data_Loading

train_dataloader, validation_dataloader = create_dataloader()

images, labels = next(iter(train_dataloader))

print("Images:", images.shape)
print("Labels:", labels.shape)

# Model

model = create_model()

model.to(device)



images = images.to(device)
labels = labels.to(device)

print("Images device:", images.device)
print("Labels device:", labels.device)

#Forward_Pass
outputs = model(images)

print("\nModel output:")
print("Shape:", outputs.shape)
print("Device:", outputs.device)
print("Values:", outputs[:5])