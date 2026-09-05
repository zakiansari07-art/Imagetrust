from datasets import load_dataset

DATASET_NAME = "TheKernel01/Tiny-GenImage"

dataset = load_dataset(DATASET_NAME)

print("Downloading Tiny-GenImage...")

dataset = load_dataset(DATASET_NAME)

print("\nDataset loaded!")
print(dataset)

for split in dataset:
    print(split)
    print(dataset[split])