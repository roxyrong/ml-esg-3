import logging
from abc import abstractmethod
import torch
from torchinfo import summary
from transformers import AutoModelForSequenceClassification

class Model(torch.nn.Module):
    def __init__(self, 
                 pretrained_model: str = None,
                 token: str = None,
                 hidden_size: int = None,
                 num_labels: int = None,
                 freeze_layers_fn = None,
                 additional_layers_fn = None,
                 device: str = None):
        super().__init__()
        self._num_labels = num_labels
        self._model_name = pretrained_model
        self.device = device
        self.pretrained = AutoModelForSequenceClassification.from_pretrained(pretrained_model, token=token).to(device)
        
        if freeze_layers_fn:
            freeze_layers_fn(self.pretrained)

        if additional_layers_fn:
            self.additional_layers = additional_layers_fn(hidden_size, num_labels).to(device)
        else:
            self.additional_layers = torch.nn.Sequential(torch.nn.Linear(hidden_size, num_labels)).to(device)

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
        
            
        
        
