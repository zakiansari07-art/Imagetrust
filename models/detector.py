import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def create_model():

    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    # Replace the original ImageNet classifier
    # 1000 classes → 1 output
    model.fc = nn.Linear(
        model.fc.in_features,
        1
    )

    return model


def create_generator_model():

    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    # Replace the original ImageNet classifier
    # 1000 classes → 1 output
    model.fc = nn.Linear(
        model.fc.in_features,
        9
    )

    return model

def create_defactify_generator_model():

    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    # Replace the original ImageNet classifier
    # 1000 classes → 1 output
    model.fc = nn.Linear(
        model.fc.in_features,
        5
    )

    return model