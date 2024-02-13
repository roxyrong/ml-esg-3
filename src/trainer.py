import json
import logging
import os
import time

import pandas as pd
import torch
from pydantic.json import pydantic_encoder
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassF1Score

from config import Config
from models import Model
from utils import AverageMeter, EarlyStopper, empty_cache


class Trainer:
    def __init__(
        self,
        model: Model,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        config: Config,
        optimizer=None,
        scheduler=None,
        early_stop: bool = True,
        device: str = "cuda",
    ) -> None:

        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = CrossEntropyLoss()
        self.early_stop = early_stop
        self.early_stopper = EarlyStopper()
        self.device = device

        self.validation_records = []

        self._create_metrics()
        self._save_model_setup()
        self._save_config()

        self.global_step = 0

    def _create_metrics(self):
        self.accuracy_metric = Accuracy(
            task="multiclass", num_classes=self.model.num_labels, top_k=1
        ).to(self.device)

        self.weighted_f1_metric = MulticlassF1Score(
            num_classes=self.model.num_labels, average="weighted"
        ).to(self.device)

        self.macro_f1_metric = MulticlassF1Score(
            num_classes=self.model.num_labels, average="macro"
        ).to(self.device)

    def fit(self, num_epochs):
        total_start_time = time.time()

        accuracy_init = 0

        for epoch in range(num_epochs):
            self.logger.info("+" * 40)
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            epoch_start_time = time.time()

            train_results = self._train_epoch()

            valid_results = self._validate_epoch(epoch)

            epoch_time = time.time() - epoch_start_time

            self._log_epoch_result(
                train_results, valid_results, epoch + 1, num_epochs, epoch_time
            )

            # save the best performance model based on validation accuracy
            if valid_results["valid_acc"] > accuracy_init:
                accuracy_init = valid_results["valid_acc"]
                self.save_checkpoint()
                self.logger.info(
                    f"New best model saved with val accuracy {accuracy_init:.4f}"
                )

            self._tensorboard_writing(epoch, train_results, valid_results)

            if self.early_stop:
                should_stop = self.early_stopper(valid_results["valid_loss"])
                if should_stop:
                    break

        total_time = time.time() - total_start_time
        self.logger.info(f"total training time: {total_time}")

        save_path = os.path.join(self.save_path, "validation_predicts.parquet")
        validation_result_df = pd.concat(self.validation_records).reset_index(drop=True)
        validation_result_df.to_parquet(save_path)

        self.summarywriter.close()
        self.logger.info("Completed Training.")

    def _train_epoch(self):
        self.logger.info("Starting training....")
        self.model.train()

        train_loss = AverageMeter()
        train_acc = AverageMeter()

        for i, (
            features,
            encoded_features,
            labels,
            encoded_labels,
            languages,
        ) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            out = self.model(
                {**encoded_features, **{"decoder_input_ids": encoded_labels}}
            )
            empty_cache(self.device)

            loss = self._compute_loss(out, labels)
            loss.backward()
            train_loss.update(loss.item(), len(out))

            acc = self.accuracy_metric(out, labels)
            train_acc.update(acc.item(), len(out))
            empty_cache(self.device)
            
            clip_grad_norm_(self.model.parameters(), 1)

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            self.global_step += 1

            # current_lr = self.scheduler.get_last_lr()[0]
            # self.logger.info(f"Current learning rate: {current_lr:.4f}")

        return {"train_loss": train_loss.avg, "train_acc": train_acc.avg}

    def _validate_epoch(self, epoch):
        self.logger.info(f"start validating...")

        self.model.eval()

        valid_loss = AverageMeter()
        valid_acc = AverageMeter()
        valid_macro_f1 = AverageMeter()
        valid_weighted_f1 = AverageMeter()

        with torch.no_grad():
            for i, (
                features,
                encoded_features,
                labels,
                encoded_labels,
                languages,
            ) in enumerate(self.valid_loader):

                out = self.model(
                    {**encoded_features, **{"decoder_input_ids": encoded_labels}}
                )

                empty_cache(self.device)

                loss = self._compute_loss(out, labels)
                acc = self.accuracy_metric(out, labels)
                macro_f1 = self.macro_f1_metric(out, labels)
                weighted_f1 = self.weighted_f1_metric(out, labels)

                valid_loss.update(loss.item(), len(out))
                valid_acc.update(acc.item(), len(out))
                valid_macro_f1.update(macro_f1.item(), len(out))
                valid_weighted_f1.update(weighted_f1.item(), len(out))

                self._append_validation_records(features, labels, out, epoch, languages)

        return {
            "valid_loss": valid_loss.avg,
            "valid_acc": valid_acc.avg,
            "valid_macro_f1": valid_macro_f1.avg,
            "valid_weighted_f1": valid_weighted_f1.avg,
        }

    def _compute_loss(self, out, labels):
        return self.criterion(out, labels)

    def _log_epoch_result(
        self, train_results, valid_results, epoch, num_epochs, epoch_time
    ):
        train_loss = train_results["train_loss"]
        train_acc = train_results["train_acc"]

        valid_loss = valid_results["valid_loss"]
        valid_acc = valid_results["valid_acc"]
        valid_macro_f1 = valid_results["valid_macro_f1"]
        valid_weighted_f1 = valid_results["valid_weighted_f1"]

        msg = f"Epoch {epoch}/{num_epochs} | Time/epoch: {round(epoch_time, 0)} seconds"
        self.logger.info(msg)

        msg = f"""global step: {self.global_step}, 
                training_loss: {train_loss:.4f},
                training_accuracy: {train_acc:.4f},
                validation_loss: {valid_loss:.4f}, 
                validation_accuracy: {valid_acc:.4f}, 
                validatoin_mcaro_f1: {valid_macro_f1:.4f}, 
                validation_weighted_f1: {valid_weighted_f1:.4f}
                """
        self.logger.info(msg)

    def _tensorboard_writing(self, epoch, train_results, valid_results):
        train_loss = train_results["train_loss"]
        train_acc = train_results["train_acc"]

        valid_loss = valid_results["valid_loss"]
        valid_acc = valid_results["valid_acc"]
        valid_macro_f1 = valid_results["valid_macro_f1"]
        valid_weighted_f1 = valid_results["valid_weighted_f1"]

        self.summarywriter.add_scalars(
            "loss/step", {"val": valid_loss, "train": train_loss}, self.global_step
        )
        self.summarywriter.add_scalars(
            "loss/epoch", {"val": valid_loss, "train": train_loss}, epoch
        )
        self.summarywriter.add_scalars(
            "val acc/epoch", {"val": valid_acc, "train": train_acc}, epoch
        )
        self.summarywriter.add_scalars("macro_f1/epoch", {"val": valid_macro_f1}, epoch)
        self.summarywriter.add_scalars(
            "weighted_f1/epoch", {"val": valid_weighted_f1}, epoch
        )

    def _save_model_setup(self):
        self.version = 0
        while True:
            ckpt_dir = "model_save"
            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)

            model_name = self.model.model_name.replace("/", "-")
            model_version = f"version-{model_name}-{self.version}"
            self.save_path = os.path.join(ckpt_dir, model_version)
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
                break
            else:
                self.version += 1
        self.summarywriter = SummaryWriter(self.save_path)

        self.logger = logging.getLogger(model_version)

        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(
            logging.FileHandler(os.path.join(self.save_path, "experiment.log"))
        )

        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.model.log_model_info(self.logger)
        self._log_trainer_info()

    def _log_trainer_info(self):
        if self.optimizer is not None:
            self.logger.info(f"optimizer: {self.optimizer}")

    def _append_validation_records(self, features, labels, out, epoch, languages):
        validation_record = pd.DataFrame(
            columns=["sentence", "label", "predict_result", "epoch", "language"]
        )
        validation_record["sentence"] = features
        validation_record["label"] = labels.detach().cpu().numpy().tolist()
        validation_record["predict_result"] = out.detach().cpu().numpy().tolist()
        validation_record["epoch"] = epoch
        validation_record["language"] = languages

        self.validation_records.append(validation_record)

    def _save_config(self):
        config_json = json.dumps(self.config.dict(), indent=4, default=pydantic_encoder)
        self.logger.info(f"Config: {config_json}")

    def save_checkpoint(self):
        save_path = os.path.join(self.save_path, "trained_model.pth")
        torch.save(self.model, save_path)
