import logging

import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
from transformers import (
    AutoModelForSequenceClassification,
    T5Config,
    T5ForConditionalGeneration,
)

from utils import check_model_type


class LayerBuilder:
    def __init__(self, hidden_size, layer_sizes, num_labels, activation, dropout):
        self.hidden_size = hidden_size
        self.layer_sizes = layer_sizes
        self.num_labels = num_labels
        self.activation_fn = self.set_activate_fn(activation)
        self.dropout = dropout

    def build(self):
        layers = []
        input_size = self.hidden_size
        for idx, size in enumerate(self.layer_sizes):
            layers.append(nn.Linear(input_size, size))
            layers.append(self.activation_fn)
            if idx < len(self.layer_sizes) - 1:
                layers.append(torch.nn.Dropout(self.dropout))
            input_size = size
        layers.append(nn.Linear(input_size, self.num_labels))
        return nn.Sequential(*layers)

    @staticmethod
    def set_activate_fn(activation):
        if activation == "LeakyReLU":
            return torch.nn.LeakyReLU()
        elif activation == "Tanh":
            return torch.nn.Tanh()
        elif activation == "Mish":
            return torch.nn.Mish()
        elif activation == "Sigmoid":
            return torch.nn.Sigmoid()
        else:
            return torch.nn.ReLU()


class Model(torch.nn.Module):
    def __init__(
        self,
        pretrained_model: str = None,
        token: str = None,
        num_labels: int = None,
        unfreeze_layers=None,
        layer_builder: LayerBuilder = None,
        device: str = None,
    ):
        super().__init__()
        self._num_labels = num_labels
        self._model_name = pretrained_model
        self.device = device
        if "t5" in pretrained_model:
            config = T5Config.from_pretrained(
                pretrained_model, output_hidden_states=True, num_labels=num_labels
            )
            self.pretrained = T5ForConditionalGeneration.from_pretrained(
                pretrained_model, config=config, token=token
            ).to(device)
        else:
            self.pretrained = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model, token=token
            ).to(device)

        self.freeze_params(unfreeze_layers)

        self.additional_layers = layer_builder.build().to(device)

    def freeze_params(self, unfreeze_layers):
        # freeze all params first
        for param in self.pretrained.parameters():
            param.requires_grad = False

        if check_model_type(self.pretrained, "bert"):
            for layer_num in unfreeze_layers:
                for param in self.pretrained.base_model.encoder.layer[
                    layer_num
                ].parameters():
                    param.requires_grad = True
        elif check_model_type(self.pretrained, "t5"):
            for layer_num in unfreeze_layers:
                for param in self.pretrained.base_model.encoder.block[
                    layer_num
                ].parameters():
                    param.requires_grad = True
                for param in self.pretrained.base_model.decoder.block[
                    layer_num
                ].parameters():
                    param.requires_grad = True

    def forward(self, encoded):
        out = self.call_pretrained(encoded)
        out = self.additional_layers(out)
        out = out.softmax(dim=1)

        return out

    def call_pretrained(self, encoded):
        if "bert" in str(type(self.pretrained)):
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            token_type_ids = encoded["token_type_ids"]
            out = self.pretrained(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True,
            )
            out = out["hidden_states"][-1][:, 0]
        elif "t5" in str(type(self.pretrained)):
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            out = self.pretrained(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=encoded["decoder_input_ids"],
                output_hidden_states=True,
            )
            out = out["encoder_last_hidden_state"]
            out = torch.mean(out, dim=1)  # average pooling
        else:
            raise KeyError("Not a known model!")
        return out

    @property
    def num_labels(self):
        return self._num_labels

    @property
    def model_name(self):
        return self._model_name

    def log_model_info(self, logger):
        msg = summary(self.pretrained)
        logger.info(msg)
        msg = f"additional layers: {self.additional_layers}"
        logger.info(msg)
