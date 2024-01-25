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

from models import Model
from utils import AverageMeter


class Trainer:
    def __init__(self, model: Model,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 optimizer = None,
                 scheduler = None,
                 device: str = "cuda") -> None:
        
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = CrossEntropyLoss()
        self.device = device
        
        self._create_metrics()
        self.save_model_setup()
        
        self.global_step = 0
        self.eval_step = 50
        
    def _create_metrics(self):
        self.accuracy_metric = Accuracy(task="multiclass", 
                            num_classes=self.model.num_labels,
                            top_k=1).to(self.device)
        
        self.weighted_f1_metric = MulticlassF1Score(num_classes=self.model.num_labels,
                                                    average="weighted").to(self.device)
        
        self.macro_f1_metric = MulticlassF1Score(num_classes=self.model.num_labels,
                                                 average="macro").to(self.device)
    
    def fit(self, num_epochs):
        logging.info("Starting training....")

        total_start_time = time.time()

        accuracy_init = 0
        
        for epoch in range(num_epochs):
            logging.info("+" * 40)
            logging.info(f"Epoch: {epoch + 1}")

            epoch_start_time = time.time()
            
            train_results = self._train_epoch()
            valid_results = self._validate()
            
            epoch_time = time.time() - epoch_start_time
            
            self._log_epoch(train_results, valid_results, epoch + 1, num_epochs, epoch_time)

            # save the best performance model based on validation accuracy
            if valid_results["valid_acc"] >= accuracy_init:
                accuracy_init = valid_results["valid_acc"]
                self.save_checkpoint()
                logging.info(f'New best model saved with val accuracy {accuracy_init:.4f}')

            self._tensorboard_writing(epoch, train_results, valid_results)
            
        total_time = time.time() - total_start_time
        logging.info(f"total training time: {total_time}")
        
        self.summarywriter.close()
        logging.info("Completed Training.")
    
    def _train_epoch(self):
        self.model.train()
        
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            out = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
            self.empty_cache()
            
            loss = self._compute_loss(out, labels)
            loss.backward()
            train_loss.update(loss.item(), len(out))

            acc = self.accuracy_metric(out, labels)
            train_acc.update(acc.item(), len(out))

            if i % self.eval_step == 0:
                logging.info(f"step {i}/{len(self.train_loader)}: train_loss: {train_loss.avg:.4f}: | train_acc: {train_acc.avg:.4f}")
            self.empty_cache()
        
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
                
            self.global_step += 1
        
        return {
            "train_loss": train_loss.avg,
            "train_acc": train_acc.avg
        }
    
    def _validate(self):
        self.model.eval()
        
        valid_loss = AverageMeter()
        valid_acc = AverageMeter()
        valid_macro_f1 = AverageMeter()
        valid_weighted_f1 = AverageMeter()
        
        with torch.no_grad():
            for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(self.valid_loader):                
                out = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
                
                self.empty_cache()
                
                loss = self._compute_loss(out, labels)
                acc = self.accuracy_metric(out, labels)
                macro_f1 = self.macro_f1_metric(out, labels)
                weighted_f1 = self.weighted_f1_metric(out, labels)
                
                valid_loss.update(loss.item(), len(out))
                valid_acc.update(acc.item(), len(out))
                valid_macro_f1.update(macro_f1.item(), len(out))
                valid_weighted_f1.update(weighted_f1.item(), len(out))
                
        return {
            "valid_loss": valid_loss.avg,
            "valid_acc": valid_acc.avg,
            "valid_macro_f1": valid_macro_f1.avg,
            "valid_weighted_f1": valid_weighted_f1.avg,
        }

    def _compute_loss(self, out, labels):
        return self.criterion(out, labels)

    def _log_epoch(self, train_results, valid_results, epoch,  num_epochs, epoch_time):
        train_loss = train_results["train_loss"]
        train_acc = train_results["train_acc"]
        
        valid_loss = valid_results["valid_loss"]
        valid_acc = valid_results["valid_acc"]
        valid_macro_f1 = valid_results["valid_macro_f1"]
        valid_weighted_f1 = valid_results["valid_weighted_f1"]

        msg = f"Epoch {epoch}/{num_epochs} | Time/epoch: {round(epoch_time, 0)} seconds"
        logging.info(msg)

        msg = f"""global step: {self.global_step}, 
                training_loss: {train_loss:.4f},
                training_accuracy: {train_acc:.4f},
                validation_loss: {valid_loss:.4f}, 
                validation_accuracy: {valid_acc:.4f}, 
                validatoin_mcaro_f1: {valid_macro_f1:.4f}, 
                validation_weighted_f1: {valid_weighted_f1:.4f}
                """
        logging.info(msg)

    def _tensorboard_writing(self, epoch, train_results, valid_results):
        train_loss = train_results["train_loss"]
        train_acc = train_results["train_acc"]
        
        valid_loss = valid_results["valid_loss"]
        valid_acc = valid_results["valid_acc"]
        valid_macro_f1 = valid_results["valid_macro_f1"]
        valid_weighted_f1 = valid_results["valid_weighted_f1"]

        self.summarywriter.add_scalars("loss/step", {"val": valid_loss, "train": train_loss}, self.global_step)
        self.summarywriter.add_scalars("loss/epoch", {"val": valid_loss, "train": train_loss}, epoch)
        self.summarywriter.add_scalars("val acc/epoch", {"val": valid_acc, "train": train_acc}, epoch)
        self.summarywriter.add_scalars("macro_f1/epoch", {"val": valid_macro_f1}, epoch)
        self.summarywriter.add_scalars("weighted_f1/epoch", {"val": valid_weighted_f1}, epoch)

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
        
        logging.basicConfig(
            filename=os.path.join(self.save_path, "experiment.log"),
            level=logging.INFO,
            format="%(asctime)s > %(message)s",
        )
        
        # Create a console handler to print in consoles
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)

        self.model.log_model_info()

    def save_checkpoint(self):
        save_path = os.path.join(self.save_path, "trained_model.pth")
        torch.save(self.model.state_dict(), save_path)

    def empty_cache(self):
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()



