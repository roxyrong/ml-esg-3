import itertools
import os

import torch
from transformers import get_linear_schedule_with_warmup

from config import Config
from datamodule import ESGDataset
from models import Model
from trainer import Trainer
from utils import empty_cache, fix_seed


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
        "config": config,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "device": config.device,
    }

    trainer = Trainer(**trainer_params)
    trainer.fit(num_epochs=config.trainer.num_epochs)


def run_grid_search(param_grid):
    param_combinations = list(
        itertools.product(*(param_grid[param] for param in param_grid))
    )

    for combination in param_combinations:
        new_params = dict(zip(param_grid.keys(), combination))
        print(f"Training with config: {new_params}")
        current_config = update_config(base_config, new_params)
        train_model(current_config)


def update_config(base_config, new_params):
    # Make a deep copy of the base config to avoid mutating the original
    current_config = base_config.model_copy(deep=True)

    # Update the configuration with new parameters
    for key, value in new_params.items():
        path = key.split(".")
        current_dict = current_config
        for part in path[:-1]:
            current_dict = getattr(current_dict, part)
        setattr(current_dict, path[-1], value)

    return current_config


# ===== Model Running Access =====
if __name__ == "__main__":
    # Load configuration file

    # config_path = "config/bert_config.yaml"
    config_path = "config/t5_config.yaml"
    
    # Train a single model
    HF_TOKEN = os.environ["HF_TOKEN"]  
    base_config = Config.from_yaml(config_path, HF_TOKEN)
    # train_model(base_config)

    param_grid = {
        "datamodule.train_path": [
            "dataset/training_dataset.parquet",
            "dataset/training_with_augmentation.parquet",
        ],
        "datamodule.batch_size": [16, 32],
        "model.unfreeze_layers": [[], [-1], [-1, -2, -3]],
        "trainer.lr": [1e-3, 1e-4, 1e-5],
    }

    run_grid_search(param_grid)
