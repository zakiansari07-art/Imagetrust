from datasets import load_dataset

DATASET_NAME = "TheKernel01/Tiny-GenImage"

dataset = load_dataset(DATASET_NAME)

print(dataset)
print(dataset["train"].cache_files)
print(dataset["validation"].cache_files)