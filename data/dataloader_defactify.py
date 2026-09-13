from typing import Tuple

from torch.utils.data import DataLoader
from datasets import load_dataset

from data.dataset_defactify import (
    DefactifyBinaryClassDataset,
    DefactifyMultiClassDataset,
)


DATASET_NAME = "Rajarshi-Roy-research/Defactify_Image_Dataset"

dataset = load_dataset(DATASET_NAME)

train_raw = dataset["train"]
val_raw = dataset["validation"]
test_raw = dataset["test"]

num_workers = 4


def create_binary_task_dataloader(
    batch_size: int = 16,
) -> Tuple[DataLoader, DataLoader]:

    train_dataset = DefactifyBinaryClassDataset(train_raw)
    validation_dataset = DefactifyBinaryClassDataset(val_raw)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, validation_dataloader


def create_binary_task_eval_dataloader(
    batch_size: int = 16,
) -> Tuple[DataLoader, DataLoader]:

    validation_dataset = DefactifyBinaryClassDataset(val_raw)
    test_dataset = DefactifyBinaryClassDataset(test_raw)

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return validation_dataloader, test_dataloader


def create_generator_task_dataloader(
    batch_size: int = 16,
) -> Tuple[DataLoader, DataLoader]:

    train_raw_filtered = train_raw.filter(lambda example: example["Label_B"] != 0)
    val_raw_filtered = val_raw.filter(lambda example: example["Label_B"] != 0)

    def id_change(example):
        example["Label_B"] = example["Label_B"] - 1
        return example

    train_new_id = train_raw_filtered.map(id_change)
    valid_new_id = val_raw_filtered.map(id_change)


    train_dataset = DefactifyMultiClassDataset(train_new_id)
    validation_dataset = DefactifyMultiClassDataset(valid_new_id)

    

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, validation_dataloader


def create_generator_task_eval_dataloader(
    batch_size: int = 16,
) -> Tuple[DataLoader, DataLoader]:

    val_raw_filtered = val_raw.filter(lambda example: example["Label_B"] != 0)
    test_raw_filtered = test_raw.filter(lambda example: example["Label_B"] != 0)
    
    def id_change(example):
        example["Label_B"] = example["Label_B"] - 1
        return example

    val_new_id = val_raw_filtered.map(id_change)
    test_new_id = test_raw_filtered.map(id_change)

    validation_dataset = DefactifyMultiClassDataset(val_new_id)
    test_dataset = DefactifyMultiClassDataset(test_new_id)

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return validation_dataloader, test_dataloader