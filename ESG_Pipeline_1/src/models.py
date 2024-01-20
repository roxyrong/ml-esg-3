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
                 device: str = None,
                 activation: str = None):
        super().__init__()
        self._num_labels = num_labels
        self._model_name = pretrained_model
        self.device = device
        self.activation = activation
        self.pretrained = AutoModelForSequenceClassification.from_pretrained(pretrained_model, token=token).to(device)
        
        if freeze_layers_fn:
            freeze_layers_fn(self.pretrained)

        if additional_layers_fn:
            self.additional_layers = torch.nn.Linear(hidden_size,256).to(device).to(device)
            self.additional_layers2 = torch.nn.Linear(256,128).to(device)
            self.additional_layers3 = torch.nn.Linear(128,64).to(device)
            self.additional_layers4 = torch.nn.Linear(64,32).to(device)
            self.additional_layers5 = torch.nn.Linear(32,3).to(device)

        else:
            self.additional_layers = torch.nn.Linear(hidden_size,256).to(device).to(device)
            self.additional_layers2 = torch.nn.Linear(256,128).to(device)
            self.additional_layers3 = torch.nn.Linear(128,64).to(device)
            self.additional_layers4 = torch.nn.Linear(64,32).to(device)
            self.additional_layers5 = torch.nn.Linear(32,3).to(device)

            self.dropout = torch.nn.Dropout(0.1)
            if self.activation == "LeakyReLU":
                self.activation = torch.nn.LeakyReLU()
            elif self.activation == "Tanh":
                self.activation = torch.nn.Tanh()
            elif self.activation == "Mish":
                self.activation = torch.nn.Mish()
            elif self.activation == "Sigmoid":
                self.activation = torch.nn.Sigmoid()
            else:
                self.activation = torch.nn.ReLU()
        self.dropout_layer = torch.nn.Dropout(0.15)

        self.fc_layer = torch.nn.Linear(1,1).to(device)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.pretrained(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              output_hidden_states=True)
        out = out["hidden_states"][-1][:, 0]
        
        out = self.additional_layers(out)
        out = self.activation(out)
        out = self.dropout_layer(out)
        out = self.additional_layers2(out)
        out = self.activation(out)
        out = self.dropout_layer(out)
        out = self.additional_layers3(out)
        out = self.activation(out)
        out = self.dropout_layer(out)
        out = self.additional_layers4(out)
        out = self.activation(out)
        out = self.dropout_layer(out)
        out = self.additional_layers5(out)
            
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
        
            
        
        
