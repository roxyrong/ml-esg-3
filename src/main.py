import os
import omegaconf
import torch
from transformers import get_linear_schedule_with_warmup
from datamodule import ESGDataset
from models import Model, LayerBuilder
from trainer import Trainer
from utils import fix_seed


def main(config):
    fix_seed(config.seed)

    # init dataset
    dm_params = {
        "train_path": config.datamodule.train_path,
        "text_col": config.datamodule.text_col,
        "label_col": config.datamodule.label_col,
        "stratify_col": config.datamodule.stratify_col,
        "tokenizer_name": config.tokenizer_name,
        "use_cls_weight": config.datamodule.use_cls_weight,
        "batch_size": config.datamodule.batch_size,
        "test_size": config.datamodule.test_size,
        "seed": config.seed,
        "device": config.device,
        "augment": config.datamodule.augment
    }

    esg_dataset = ESGDataset(**dm_params)
    train_loader = esg_dataset.train_dataloader()
    valid_loader = esg_dataset.valid_dataloader()

    # init model
    def custom_freeze_layers(model):
        for param in model.parameters():
            param.requires_grad = False
            
    layer_params = {
        "hidden_size": config.model.hidden_size,
        "layer_sizes": config.model.layer_sizes,
        "activation": config.model.activation,
        "num_labels": config.model.num_labels,
        "dropout": config.model.dropout
    }
            
    layer_builder = LayerBuilder(**layer_params)
    
    model_params = {
        "pretrained_model":config.pretrained_model,
        "token": config.hf_token,
        "num_labels": config.model.num_labels,
        "freeze_layers_fn": custom_freeze_layers,
        "layer_builder": layer_builder,
        "device": config.device,
    }
    
    model = Model(**model_params)

    # init_trainer
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=config.trainer.lr, 
                                weight_decay=config.trainer.weight_decay)
    
    num_training_steps = config.trainer.num_epochs * len(train_loader)  
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=int(num_training_steps * config.trainer.warm_up_step), 
                                                num_training_steps=num_training_steps)
    
    trainer_params =  {
        "model": model,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "device": config.device
    }

    trainer = Trainer(**trainer_params)
    trainer.fit(num_epochs=config.trainer.num_epochs)


# ===== Model Running Access =====
if __name__ == "__main__":
    # Load configuration file
    config_path = "config/config.yaml"
    config = omegaconf.OmegaConf.load(config_path)
    # Append huggingface token
    HF_TOKEN = os.environ["HF_TOKEN"]
    config.hf_token = HF_TOKEN
    # Run the whole pipeline
    main(config)

