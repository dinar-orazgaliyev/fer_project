import torch
import torch.nn as nn
from functools import partial
from tqdm import tqdm
import torch.optim as optim
import logging

from torchvision.utils import make_grid
from .base_trainer import BaseTrainer
from models.convnetfer_model import ConvNetFer

class CNNTrainer(BaseTrainer):

    def __init__(self, config, log_dir, train_loader, eval_loader=None):
        super().__init__(config,log_dir)
        config['model_args']['activation'] = getattr(nn, config['model_args']['activation'])  # e.g., 'nn.ReLU' → nn.ReLU
        config['model_args']['norm_layer'] = getattr(nn, config['model_args']['norm_layer'])
        self.model = eval(config['model_name'])(**config['model_args'])
        self.model.to(self._device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.criterion = getattr(nn,config['train_args']['criterion'])()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def weights_init(self, m):
        """
        Initializes the model weights! Must be used with .apply of an nn.Module so that it works recursively!
        """
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 1e-2)
            nn.init.normal_(m.bias, 0.0, 1e-2)

    def _train_epoch(self):
        running_loss = 0.0  # initialize before the loop
        correct = 0
        total = 0
        # # Set model to train mode
        self.model.train()
        

        self.logger.info(f"Start Training Epoch {self.current_epoch}/{self.epochs}, lr = {self.config['train_args']['optim_args']['lr']}")
        pbar = tqdm(total=len(self.train_loader) * self.train_loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        for batch_idx, (labels,images) in enumerate(self.train_loader):
            labels = labels.to(self._device)
            images = images.to(self._device).float()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=loss.item())

        train_loss = running_loss / total
        train_acc = correct / total
        self.model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for labels, inputs in self.eval_loader:
                inputs = inputs.float()  # Ensure inputs are float
                inputs, labels = inputs.to(self._device), labels.to(self._device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss /= total
        val_acc = correct / total

        print(f"Epoch {self.epoch+1}/{self.epochs} "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        
        return {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }

        # """
        # Training logic for an epoch. Only takes care of doing a single training loop.

        # :return: A dict that contains average loss and metric(s) information in this epoch.
        # """

        # # Set model to train mode
        # self.model.train()
        # self.train_metrics.reset()

        # self.logger.debug(f"==> Start Training Epoch {self.current_epoch}/{self.epochs}, lr={self.optimizer.param_groups[0]['lr']:.6f} ")

        # pbar = tqdm(total=len(self._train_loader) * self._train_loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        # for batch_idx, (images, labels) in enumerate(self._train_loader):

        #     images = images.to(self._device)
        #     labels = labels.to(self._device)


        #     output = self.model(images)
        #     loss = self.criterion(output, labels)

        #     loss.backward()
        #     self.optimizer.step()
        #     self.optimizer.zero_grad()

        #     if self.writer is not None:
        #         self.writer.set_step((self.current_epoch - 1) * len(self._train_loader) + batch_idx)
            
        #     # Update all the train_metrics with new values.
        #     self.train_metrics.update('loss', loss.item())
        #     for metric_key, metric_func in self.metric_functions.items():
        #         self.train_metrics.update(metric_key, metric_func.compute(output, labels))

        #     pbar.set_description(f"Train Epoch: {self.current_epoch} Loss: {loss.item():.4f}")

        #     if self.writer is not None and batch_idx % self.log_step == 0:
        #         self.writer.add_image('input_train', make_grid(images.cpu(), nrow=8, normalize=True))

        #     pbar.update(self._train_loader.batch_size)

        # log_dict = self.train_metrics.result()
        # pbar.close()
        # self.lr_scheduler.step()

        # self.logger.debug(f"==> Finished Epoch {self.current_epoch}/{self.epochs}.")
        
        # return log_dict

    @torch.no_grad()
    def evaluate(self, loader=None):
        pass
        # """
        # Evaluate the model on the val_loader given at initialization

        # :param loader: A Dataloader to be used for evaluatation. If not given, it will use the 
        # self._eval_loader that's set during initialization..
        # :return: A dict that contains metric(s) information for validation set
        # """
        # if loader is None:
        #     assert self._eval_loader is not None, 'loader was not given and self._eval_loader not set either!'
        #     loader = self._eval_loader

        # self.model.eval()
        # self.eval_metrics.reset()

        # self.logger.debug(f"++> Evaluate at epoch {self.current_epoch} ...")

        # pbar = tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        # for batch_idx, (images, labels) in enumerate(loader):
            
        #     images = images.to(self._device)
        #     labels = labels.to(self._device)

        #     output = self.model(images)
        #     loss = self.criterion(output, labels)

        #     if self.writer is not None: self.writer.set_step((self.current_epoch - 1) * len(loader) + batch_idx, 'valid')
        #     self.eval_metrics.update('loss', loss.item())
        #     for metric_key, metric_func in self.metric_functions.items():
        #         self.eval_metrics.update(metric_key, metric_func.compute(output, labels))

        #     pbar.set_description(f"Eval Loss: {loss.item():.4f}")
        #     if self.writer is not None:
        #         self.writer.add_image('input_valid', make_grid(images.cpu(), nrow=8, normalize=True))

        #     pbar.update(loader.batch_size)

        # # add histogram of model parameters to the tensorboard (This can be very slow for big models.)
        # # if self.writer is not None:
        # #     for name, p in self.model.named_parameters():
        # #         self.writer.add_histogram(name, p, bins='auto')

        # pbar.close()
        # self.logger.debug(f"++> Finished evaluating epoch {self.current_epoch}.")
        
        # return self.eval_metrics.result()