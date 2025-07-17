import torch
import torch.nn as nn
from functools import partial
from tqdm import tqdm
import torch.optim as optim
import logging

from torchvision.utils import make_grid
from .base_trainer import BaseTrainer
from models.convnetfer_model import ConvNetFer


logger = logging.getLogger(__name__)
class CNNTrainer(BaseTrainer):

    def __init__(self, config,  train_loader, eval_loader=None):
        self.log_dir = config['log_dir']
        super().__init__(config,self.log_dir)
        if isinstance(config['model_args']['activation'], str):
            config['model_args']['activation'] = getattr(nn, config['model_args']['activation'])
        if isinstance(config['model_args']['norm_layer'], str):
            config['model_args']['norm_layer'] = getattr(nn, config['model_args']['norm_layer'])
        self.model = eval(config['model_name'])(**config['model_args'])
        self.model.to(self._device)
        self.model.apply(self.weights_init)
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
        epoch_progress = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}/{self.epochs}", leave=True, unit="batch")
        
        for batch_idx, (labels,images) in enumerate(epoch_progress):
            labels = labels.to(self._device)
            images = images.to(self._device).float()
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            epoch_progress.set_postfix(loss=f"{loss.item():.4f}")

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
        
        logger.info(f"Epoch {self.epoch+1}/{self.epochs} "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        return {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }


    @torch.no_grad()
    def evaluate(self, loader=None):
        pass