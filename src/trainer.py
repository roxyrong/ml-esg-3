import gc
import time
import logging

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from models import Model

class Trainer:
    def __init__(self, model: Model,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 lr: float = None,
                 weight_decay: float = None,
                 device: str = "mps") -> None:
        
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.optimizer = self._configure_optimizer(
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = CrossEntropyLoss()
        self.scheduler = None
        self.device = device
        
    
    def _configure_optimizer(self, lr, weight_decay):
        return torch.optim.AdamW(self.model.parameters(), 
                                 lr=lr, 
                                 weight_decay=weight_decay)
        
    def _configure_scheduler(self, num_epochs):
        num_training_steps = num_epochs * len(self.train_loader)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                         num_warmup_steps=num_training_steps * 0.1, 
                                                        num_training_steps=num_training_steps)
    
    def fit(self, num_epochs):
        self._configure_scheduler(num_epochs)
        
        total_start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # train
            train_loss = self._train()

            # validate
            valid_loss = self._validate()
            
            epoch_time = time.time() - epoch_start_time
            
            self._logger(
                train_loss, 
                valid_loss, 
                epoch+1, 
                num_epochs, 
                epoch_time, 
                **self.logger_kwargs
            )
            
        total_time = time.time() - total_start_time
    
    def _train(self):
        self.model.train()
        
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            out = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
            
            self.empty_cache()
            
            loss = self._compute_loss(out, labels)
            loss.backward()
            
            self.empty_cache()
        
            self.optimizer.step()
            self.scheduler.step()
        
        return loss.item()
    
    def _validate(self):
        self.model.eval()
        
        with torch.no_grad():
            for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(self.valid_loader):                
                out = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
                
                loss = self._compute_loss(out, labels)

        return loss.item()
    
    def _compute_loss(self, out, labels):
        loss = self.criterion(out, labels)
        return loss
    
    def _logger(self, train_loss, valid_loss, epoch,  num_epochs, epoch_time, show=True, update_step=1):
        if show:
            msg = f"Epoch {epoch}/{num_epochs} | Train loss: {train_loss}" 
            msg = f"{msg} | Validation loss: {valid_loss}"
            msg = f"{msg} | Time/epoch: {round(epoch_time, 5)} seconds"
            logging.info(msg)
    
    def save_checkpoint(self):
        return
    
    def empty_cache(self):
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
