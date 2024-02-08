from typing import List

import omegaconf
from pydantic import BaseModel

from models import LayerBuilder


class DataModuleConfig(BaseModel):
    train_path: str
    valid_path: str
    feature_col: str
    language_col: str
    label_col: str
    stratify_col: str
    max_length: int
    use_cls_weight: bool
    batch_size: int
    test_size: float


class ModelConfig(BaseModel):
    hidden_size: int
    layer_sizes: List[int]
    unfreeze_layers: List[int]
    num_labels: int
    dropout: float
    activation: str


class TrainerConfig(BaseModel):
    num_epochs: int
    lr: float
    weight_decay: float
    warm_up_step: float


class Config(BaseModel):
    hf_token: str
    seed: int
    pretrained_model: str
    tokenizer_name: str
    device: str
    datamodule: DataModuleConfig
    model: ModelConfig
    trainer: TrainerConfig

    @classmethod
    def from_yaml(cls, config_path: str, hf_token: str):
        config = omegaconf.OmegaConf.load(config_path)
        config.hf_token = hf_token
        return cls(**config)

    def datamodule_params(self):
        return {
            "train_path": self.datamodule.train_path,
            "valid_path": self.datamodule.valid_path,
            "feature_col": self.datamodule.feature_col,
            "language_col": self.datamodule.language_col,
            "label_col": self.datamodule.label_col,
            "stratify_col": self.datamodule.stratify_col,
            "tokenizer_name": self.tokenizer_name,
            "use_cls_weight": self.datamodule.use_cls_weight,
            "batch_size": self.datamodule.batch_size,
            "test_size": self.datamodule.test_size,
            "seed": self.seed,
            "device": self.device,
        }

    def layer_params(self):
        return {
            "hidden_size": self.model.hidden_size,
            "layer_sizes": self.model.layer_sizes,
            "activation": self.model.activation,
            "num_labels": self.model.num_labels,
            "dropout": self.model.dropout,
        }

    def model_params(self):
        layer_params = self.layer_params()
        layer_builder = LayerBuilder(**layer_params)
        return {
            "pretrained_model": self.pretrained_model,
            "token": self.hf_token,
            "num_labels": self.model.num_labels,
            "unfreeze_layers": self.model.unfreeze_layers,
            "layer_builder": layer_builder,
            "device": self.device,
        }
