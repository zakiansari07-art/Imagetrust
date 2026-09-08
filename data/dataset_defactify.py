import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DefactifyBinaryClassDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225],
                                 )
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample =  self.dataset[idx]
        image = sample["Image"].convert("RGB")
        label = sample["Label_A"]
        image = self.transform(image)
        return image, label
        


class DefactifyMultiClassDataset(Dataset):

    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        sample = self.dataset[idx]

        image = sample["Image"].convert("RGB")

        

        generator = sample["Label_B"]

        image = self.transform(image)

        return (
            image,
            generator
        )

class Defactify_Data_Augmentation(Dataset):

    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        sample = self.dataset[idx]

        image = sample["image"].convert("RGB")

        label = sample["label"]

        generator = sample["generator"]

        image = self.transform(image)

        return (
            image,
            label,
            generator
        )