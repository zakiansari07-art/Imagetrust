import torch
from data.dataloader import create_generator_evaluation_dataloader
from models.detector import create_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device: ",device)

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

model = create_model()


checkpoint_name = "models/checkpoint/resnet_18_ai_detector_best.pth"

checkpoint = torch.load(checkpoint_name, map_location=device)



model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)

model.eval()

validation_dataloader = create_generator_evaluation_dataloader()

images, labels, generators = next(iter(validation_dataloader))

images = images.to(device)
labels = labels.to(device)
generators = generators.to(device)

print("Images:", images.shape)
print("Labels:", labels)
print("Generators:", generators)

with torch.no_grad():
    logits = model(images)
    
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).float()

print("Labels:       ", labels.cpu().numpy())
print("Generators:   ", generators.cpu().numpy())
print("Probabilities:", probabilities.squeeze().cpu().numpy())
print("Predictions:  ", predictions.squeeze().cpu().numpy())
