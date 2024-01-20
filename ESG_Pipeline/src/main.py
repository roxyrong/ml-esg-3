import os
from dotenv import load_dotenv
import omegaconf
import torch
from transformers import get_linear_schedule_with_warmup
from datamodule import ESGDataset
from models import Model
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
        "device": config.device
    }

    esg_dataset = ESGDataset(**dm_params)
    train_loader = esg_dataset.train_dataloader()
    valid_loader = esg_dataset.valid_dataloader()

    # init model
    def custom_freeze_layers(model):
        for param in model.parameters():
            param.requires_grad = False
    
    model_params = {
        "pretrained_model":config.pretrained_model,
        "token": config.hf_token,
        "hidden_size": config.model.hidden_size,
        "num_labels": config.model.num_labels,
        "freeze_layers_fn": custom_freeze_layers,
        "device": config.device,
        "activation": config.model.activation,
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


if __name__ == "__main__":
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    config_path = "./config.yaml"
    config = omegaconf.OmegaConf.load(config_path)

    config.hf_token = HF_TOKEN

    main(config)

