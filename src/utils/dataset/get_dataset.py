from datasets import load_dataset


def get_dataset(file_path, split="train"):
    dataset = load_dataset("json", data_files=file_path, split=split)
    return dataset
