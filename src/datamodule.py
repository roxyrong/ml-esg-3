import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer
import re


def collate_fn(data, text_col, label_col, tokenizer, max_length, device):
    sentences = [i[text_col] for i in data]
    labels = torch.tensor([i[label_col] for i in data]).to(device)
    data = tokenizer.batch_encode_plus(sentences,
                                    truncation=True,
                                    padding='max_length',
                                    max_length=max_length,
                                    return_tensors='pt',
                                    return_length=True)

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    token_type_ids = data['token_type_ids'].to(device)
    return input_ids, attention_mask, token_type_ids, labels


class ESGDataset:
    def __init__(self, 
                 train_path: str = None,
                 text_col: str = None,
                 label_col: str = None,
                 stratify_col: str = None,
                 tokenizer_name: str = None,
                 max_length: int = 512,
                 use_cls_weight: bool = True,
                 batch_size: int = 32,
                 test_size: float = 0.2,
                 seed: int = 3407,
                 device: str = "cuda",
                 augment: bool = False) -> None:
        
        self.train_path = train_path
        self.text_col = text_col
        self.label_col = label_col
        self.stratify_col = stratify_col
        self.max_length = max_length
        self.use_cls_weight = use_cls_weight
        self.batch_size = batch_size
        self.test_size = test_size
        self.seed = seed
        self.device = device
        self.augment = augment
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.setup()


    def setup(self):
        # import the parquet
        self.df = pd.read_parquet(self.train_path).loc[:100]

        # === using augmentation
        if self.augment:
            print("Augmented Dataset detected...")
            # segment the dataset into two parts
            self.augmented_dataset = self.df.loc[self.df["source"] == "gpt"]
            self.df = self.df.loc[self.df["source"] != "gpt"]
            # convert two datasets to Datasets
            esg_dataset = Dataset.from_pandas(self.df, preserve_index=True)
            self.augmented_dataset = Dataset.from_pandas(self.augmented_dataset, preserve_index=True)
            # train test split
            self.train_dataset, self.valid_dataset = self._train_test_split(esg_dataset)
            if self.use_cls_weight:
                # concatenate dataset
                self.augmented_dataset = self.augmented_dataset.class_encode_column(self.label_col)
                self.train_dataset = concatenate_datasets([self.train_dataset, self.augmented_dataset])
                self.train_dataset = self.resample_train_dataset()
            else:
                # concatenate dataset
                self.augmented_dataset = self.augmented_dataset.class_encode_column(self.label_col)
                self.train_dataset = concatenate_datasets([self.train_dataset, self.augmented_dataset])
        else:
            # filter out augmented sentences
            self.df = self.df.loc[self.df["source"] != "gpt"]
            # convert to Datasets
            esg_dataset = Dataset.from_pandas(self.df, preserve_index=True)
            # train test split
            self.train_dataset, self.valid_dataset = self._train_test_split(esg_dataset)
            if self.use_cls_weight:
                self.train_dataset = self.resample_train_dataset()


    
    def _filter_language(self):
        #TODO: filter original languages
        self.df = self.df.loc[self.df["language"] != "English"]
        return self.df
    
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.batch_size,
                          collate_fn=lambda x: collate_fn(x, self.text_col, self.label_col, self.tokenizer, self.max_length, self.device),
                          shuffle=True,
                          drop_last=True)
        
    def valid_dataloader(self):
        return DataLoader(dataset=self.valid_dataset,
                          batch_size=self.batch_size,
                          collate_fn=lambda x: collate_fn(x, self.text_col, self.label_col, self.tokenizer, self.max_length, self.device))
    
    def _train_test_split(self, esg_dataset: Dataset):
        if self.use_cls_weight:
            esg_dataset = esg_dataset.class_encode_column(self.label_col)
            esg_dataset = esg_dataset.train_test_split(test_size=self.test_size,
                                                       stratify_by_column=self.stratify_col,
                                                       seed=self.seed)
        else:
            esg_dataset = esg_dataset.train_test_split(test_size=self.test_size)
        
        return esg_dataset["train"], esg_dataset["test"]
            
    def resample_train_dataset(self):
        # upsample the larger classes and downsample the smaller classes by apply weights
        train_idx = sorted(self.train_dataset['__index_level_0__'])
        total_count = len(train_idx)
        class_labels = set(self.train_dataset[self.label_col])
        num_classes = len(class_labels)
        desired_sample_size = total_count // num_classes

        resampled_datasets = []

        for class_label in class_labels:
            class_dataset = self.train_dataset.filter(lambda x: x[self.label_col] == class_label)
            current_class_size = len(class_dataset)
            
            if current_class_size > desired_sample_size:
                shuffled_dataset = class_dataset.shuffle(seed=self.seed)
                downsampled = shuffled_dataset.select(range(desired_sample_size)) 
                resampled_datasets.append(downsampled)
            else:
                random_indices = [random.randint(0, current_class_size - 1) for _ in range(desired_sample_size)]
                upsampled = class_dataset.select(random_indices)
                resampled_datasets.append(upsampled)
                            
        return concatenate_datasets(resampled_datasets)
        
        
        



