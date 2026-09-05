from torch.utils.data import DataLoader
from datasets import load_dataset
from data.dataset import Imagedataset, GeneratorEvaluationDataset, Data_Augmentation_train

DATASET_NAME = "TheKernel01/Tiny-GenImage"

num_workers = 5

def create_dataloader(batch_size: int = 16) -> DataLoader:

    hf_dataset = load_dataset(DATASET_NAME)

    train_dataset = Imagedataset(hf_dataset["train"])

    validation_dataset = Imagedataset(hf_dataset["validation"])


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, validation_dataloader

def create_generator_evaluation_dataloader(
    batch_size: int = 16
) -> DataLoader:

    hf_dataset = load_dataset(DATASET_NAME)

    validation_dataset = GeneratorEvaluationDataset(
        hf_dataset["validation"]
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return validation_dataloader


def create_generator_dataloader(batch_size: int = 16) -> DataLoader:

    hf_dataset = load_dataset(DATASET_NAME)

    train_dataset = GeneratorEvaluationDataset(hf_dataset["train"])

    validation_dataset = GeneratorEvaluationDataset(hf_dataset["validation"])


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dataloader, validation_dataloader



