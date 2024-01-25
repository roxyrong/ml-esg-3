import logging
import torch
import torch.nn as nn
from torchinfo import summary
from transformers import AutoModelForSequenceClassification

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
    def __init__(self, 
                 pretrained_model: str = None,
                 token: str = None,
                 num_labels: int = None,
                 freeze_layers_fn = None,
                 layer_builder: LayerBuilder = None,
                 device: str = None):
        super().__init__()
        self._num_labels = num_labels
        self._model_name = pretrained_model
        self.device = device
        self.pretrained = AutoModelForSequenceClassification.from_pretrained(pretrained_model, token=token).to(device)
        
        if freeze_layers_fn:
            freeze_layers_fn(self.pretrained)
            
        self.additional_layers = layer_builder.build().to(device)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.pretrained(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              output_hidden_states=True)
        out = out["hidden_states"][-1][:, 0]
        out = self.additional_layers(out)
        out = out.softmax(dim=1)
        
        return out
    
    @property
    def num_labels(self):
        return self._num_labels
    
    @property
    def model_name(self):
        return self._model_name
    
    def log_model_info(self):
        msg = summary(self.pretrained)
        logging.info(msg)
        msg = f"additional layers: {self.additional_layers}"
        logging.info(msg)
        
            
        
        
