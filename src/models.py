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
                 allow_finetune: bool = False, #TODO
                 device: str = None):
        super().__init__()
        self._num_labels = num_labels
        self._model_name = pretrained_model
        self.device = device
        self.pretrained = AutoModelForSequenceClassification.from_pretrained(pretrained_model, token=token).to(device)
        self.freeze_layers()
        self.fcs = self._fcs(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.pretrained(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              output_hidden_states=True)
        out = out["hidden_states"][-1][:, 0]
        
        for fc in self.fcs:
            out = fc(out)
            
        out = out.softmax(dim=1)
        
        return out
    
    @property
    def num_labels(self):
        return self._num_labels
    
    @property
    def model_name(self):
        return self._model_name
    
    def _fcs(self, hidden_size, num_labels):
        #TODO: add more customization to pass the number of layers and dropout
        return [torch.nn.Linear(hidden_size, num_labels).to(self.device)]
    
    @abstractmethod
    def freeze_layers(self):
        pass


class MultiLingualBERT(Model):
    def __init__(self, 
                 pretrained_model: str = None,
                 token: str = None, 
                 hidden_size: int = None,
                 num_labels: int = None,
                 device: str = "mps"):
        super().__init__(pretrained_model=pretrained_model, 
                         token=token, 
                         hidden_size=hidden_size, 
                         num_labels=num_labels, 
                         device=device)
        
    def freeze_layers(self):
        for layer in self.pretrained.bert.encoder.layer[:6]:
            for param in layer.parameters():
                param.requires_grad = False
        
            
        
        
