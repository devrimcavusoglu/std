import datasets
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    """Dataset for classification."""

    def __init__(self, data_root, split, pipeline=None):
        self.data_source = datasets.load_dataset(data_root, split=split)
        self.pipeline = pipeline

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        instance = self.data_source[idx]
        if self.pipeline is not None:
            instance["image"] = self.pipeline(instance["image"])

        return instance["image"], instance["label"]
