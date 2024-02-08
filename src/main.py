import os
import itertools

import torch
from transformers import get_linear_schedule_with_warmup

from config import Config
from datamodule import ESGDataset
from models import Model
from trainer import Trainer
from utils import fix_seed, empty_cache


def train_model(config: Config):
    empty_cache(config.device)
    fix_seed(config.seed)

    # init dataset
    dm_params = config.datamodule_params()
    esg_dataset = ESGDataset(**dm_params)
    train_loader = esg_dataset.train_dataloader()
    valid_loader = esg_dataset.valid_dataloader()

    # init model
    model_params = config.model_params()
    model = Model(**model_params)

    # init_trainer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.trainer.lr,
        weight_decay=config.trainer.weight_decay,
    )

    num_training_steps = config.trainer.num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * config.trainer.warm_up_step),
        num_training_steps=num_training_steps,
    )

    trainer_params = {
        "model": model,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "device": config.device,
    }

    trainer = Trainer(**trainer_params)
    trainer.fit(num_epochs=config.trainer.num_epochs)


def run_grid_search(param_grid):
    param_grid_keys = param_grid.keys()
    print(param_grid_keys)
    param_combinations = list(
        itertools.product(*(param_grid[param] for param in param_grid))
    )

    for combination in param_combinations:
        new_params = dict(zip(param_grid_keys, combination))
        print(f"Training with config: {new_params}")
        current_config = base_config.copy(deep=True)
        current_config.datamodule.train_path = new_params["train_path"]
        current_config.pretrained_model = new_params["pretrained_model"]
        current_config.tokenizer_name = new_params["pretrained_model"]
        current_config.trainer.lr = new_params["lr"]
        current_config.model.unfreeze_layers = new_params["unfreeze_layers"]
        train_model(current_config)


# ===== Model Running Access =====
if __name__ == "__main__":
    # Load configuration file
    HF_TOKEN = os.environ["HF_TOKEN"]
    config_path = "config/config.yaml"
    
    # Train a single model
    base_config = Config.from_yaml(config_path, HF_TOKEN)
    train_model(base_config)

    # Grid Search
    # param_grid = {
    #     'train_path': ['dataset/training_dataset.parquet', 'dataset/training_with_augmentation.parquet'],
    #     'pretrained_model': ['nbroad/ESG-BERT', 'microsoft/deberta-v3-base'],
    #     'lr': [1e-3, 1e-4, 1e-5],
    #     'batch_size': [16, 32],
    #     'unfreeze_layers': [[], [11], [9, 10, 11]]
    # }

    # run_grid_search(param_grid)
