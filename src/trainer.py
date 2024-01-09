import os
import gc
import time
import logging

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassF1Score
from transformers import get_linear_schedule_with_warmup

from models import Model

class Trainer:
    def __init__(self, model: Model,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 optimizer = None,
                 scheduler = None,
                 monitor: str = None,
                 device: str = "mps") -> None:
        
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = CrossEntropyLoss()
        self.device = device
        
        self.create_metrics()
        self.save_model_setup()
        
        self.global_step = 0
        self.eval_step = 25
    
    def fit(self, num_epochs):        
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
            
            self.summarywriter.add_scalars(
                "loss/epoch", {"val": valid_loss, "train": train_loss}, 
                epoch)
            
        total_time = time.time() - total_start_time
        self.summarywriter.close()
    
    def _train(self):
        self.model.train()
        
        train_loss = []
        
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(self.train_loader):
            
            self.optimizer.zero_grad()
            
            out = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
            
            self.empty_cache()
            
            loss = self._compute_loss(out, labels)
            loss.backward()
            train_loss.append(loss.item())
            
            if i % self.eval_step == 0:
                logging.info(f"step {i}/{len(self.train_loader)}: train_loss: {torch.mean(train_loss)}")
            
            self.empty_cache()
        
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
        
        return torch.mean(train_loss)
    
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
            
    def create_metrics(self):
        self.acc = Accuracy(task="multiclass", 
                            num_classes=self.model.num_labels,
                            top_k=1).to(self.device)
        
        self.weighted_f1_metric = MulticlassF1Score(num_classes=self.model.num_labels,
                                                    average="weighted").to(self.device)
        
        self.macro_f1_metric = MulticlassF1Score(num_classes=self.model.num_labels,
                                                 average="macro").to(self.device)
        
    def save_model_setup(self):
        self.version = 0
        while True:
            ckpt_dir = "model_save"
            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)

            self.save_path = os.path.join(ckpt_dir, f"version-{self.model.model_name}-{self.version}")
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
                break
            else:
                self.version += 1
        self.summarywriter = SummaryWriter(self.save_path)
    
    def save_checkpoint(self):
        self.model.module.save_pretrained()
    
    def empty_cache(self):
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
