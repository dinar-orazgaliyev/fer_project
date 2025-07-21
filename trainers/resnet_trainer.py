import torch
import torch.nn as nn
from functools import partial
from tqdm import tqdm
import torch.optim as optim
import logging

from torchvision.utils import make_grid
from .base_trainer import BaseTrainer
from models.resnetfer_model import ResNetFer

from torch.utils.tensorboard import SummaryWriter
from utils.earlystopping import EarlyStopping

logger = logging.getLogger(__name__)
class ResNetTrainer(BaseTrainer):
    def __init__(self, config, train_loader, eval_loader=None):
        # Call the parent's __init__ to set up logger, writer, etc.
        super().__init__(config, config['log_dir'])
        self.log_dir = config['log_dir']

        # --- 1. Model Initialization ---
        # This call now perfectly matches the new ResNetFer.__init__ signature.
        self.model = ResNetFer(
            num_classes=config["model_args"]["num_classes"],
            dropout=config["model_args"]["dropout"],
        ).to(self._device)

        # --- 2. Define Hyperparameters ---
        self.lr_head = 2e-4
        self.lr_base = 2e-5
        self.weight_decay = 5e-5

        # --- 3. Create the Single, Differential Optimizer ---
        # This will work because the model now has the correct attributes.
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.model.classifier.parameters(),
                    "lr": self.lr_head,
                },
                {
                    "params": self.model.feature_extractor.parameters(),
                    "lr": self.lr_base,
                },
            ],
            weight_decay=self.weight_decay,
        )
        
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer,
        #     mode='min',      # Reduce on minimum validation loss
        #     factor=0.2,      # Reduce LR by a factor of 5 (1/5 = 0.2)
        #     patience=5       # Wait 5 epochs of no improvement before reducing
        # )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs, # The number of epochs to complete one cosine cycle
            eta_min=1e-7        # The minimum learning rate to decay to
        )

        # --- 4. DataLoaders and Criterion ---
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # Loss function
        # self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        criterion_name = config['train_args'].get('criterion', 'CrossEntropyLoss')
        self.criterion = getattr(nn, criterion_name)()

    def _train_epoch(self):
        running_loss = 0.0  # initialize before the loop
        correct = 0
        total = 0
        # # Set model to train mode
        self.model.train()
        

        self.logger.info(f"Start Training Epoch {self.current_epoch}/{self.epochs}, lr_head = {self.lr_head}, lr_base = {self.lr_base}")
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
        self.scheduler.step()

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