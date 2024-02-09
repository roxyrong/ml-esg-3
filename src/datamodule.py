import random

import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5Tokenizer


def collate_fn(
    data, feature_col, label_col, language_col, tokenizer, max_length, device
):
    features = [i[feature_col] for i in data]
    labels = torch.tensor([i[label_col] for i in data]).to(device)
    languages = [i[language_col] for i in data]
    encoded_features = tokenizer.batch_encode_plus(
        features,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    encoded_labels = tokenizer.batch_encode_plus(
        [str(i[label_col]) for i in data],
        truncation=True,
        padding="max_length",
        max_length=2,
        return_tensors="pt",
    ).input_ids
    encoded_features = encoded_features.to(device)
    encoded_labels = encoded_labels.to(device)
    return features, encoded_features, labels, encoded_labels, languages


class ESGDataset:
    def __init__(
        self,
        train_path: str = None,
        valid_path: str = None,
        feature_col: str = None,
        label_col: str = None,
        language_col: str = None,
        stratify_col: str = None,
        tokenizer_name: str = None,
        max_length: int = 400,
        use_cls_weight: bool = True,
        batch_size: int = 32,
        test_size: float = 0.2,
        seed: int = 3407,
        device: str = "cuda",
        augment: bool = False,
    ) -> None:

        self.train_path = train_path
        self.valid_path = valid_path
        self.feature_col = feature_col
        self.label_col = label_col
        self.language_col = language_col
        self.stratify_col = stratify_col
        self.max_length = max_length
        self.use_cls_weight = use_cls_weight
        self.batch_size = batch_size
        self.test_size = test_size
        self.seed = seed
        self.device = device
        self.augment = augment

        if "t5" in tokenizer_name:
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.setup()

    def setup(self):
        # import the parquet
        self.train_df = pd.read_parquet(self.train_path)
        self.valid_df = pd.read_parquet(self.valid_path)

        self.train_dataset = Dataset.from_pandas(self.train_df, preserve_index=True)
        self.valid_dataset = Dataset.from_pandas(self.valid_df, preserve_index=True)

        self.train_dataset = self.train_dataset.class_encode_column(self.label_col)
        self.valid_dataset = self.valid_dataset.class_encode_column(self.label_col)

        if self.use_cls_weight:
            self.train_dataset = self.resample_train_dataset()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda data: collate_fn(
                data=data,
                feature_col=self.feature_col,
                language_col=self.language_col,
                label_col=self.label_col,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                device=self.device,
            ),
            shuffle=True,
            drop_last=True,
        )

    def valid_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda data: collate_fn(
                data=data,
                feature_col=self.feature_col,
                language_col=self.language_col,
                label_col=self.label_col,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                device=self.device,
            ),
        )

    def _train_test_split(self, esg_dataset: Dataset):
        if self.use_cls_weight:
            esg_dataset = esg_dataset.class_encode_column(self.label_col)
            esg_dataset = esg_dataset.train_test_split(
                test_size=self.test_size,
                stratify_by_column=self.stratify_col,
                seed=self.seed,
            )
        else:
            esg_dataset = esg_dataset.train_test_split(test_size=self.test_size)

        return esg_dataset["train"], esg_dataset["test"]

    def resample_train_dataset(self):
        # upsample the larger classes and downsample the smaller classes by apply weights
        train_idx = sorted(self.train_dataset["__index_level_0__"])
        total_count = len(train_idx)
        class_labels = set(self.train_dataset[self.label_col])
        num_classes = len(class_labels)
        desired_sample_size = total_count // num_classes

        resampled_datasets = []

        for class_label in class_labels:
            class_dataset = self.train_dataset.filter(
                lambda x: x[self.label_col] == class_label
            )
            current_class_size = len(class_dataset)

            if current_class_size > desired_sample_size:
                shuffled_dataset = class_dataset.shuffle(seed=self.seed)
                downsampled = shuffled_dataset.select(range(desired_sample_size))
                resampled_datasets.append(downsampled)
            else:
                random_indices = [
                    random.randint(0, current_class_size - 1)
                    for _ in range(desired_sample_size)
                ]
                upsampled = class_dataset.select(random_indices)
                resampled_datasets.append(upsampled)

        return concatenate_datasets(resampled_datasets)
