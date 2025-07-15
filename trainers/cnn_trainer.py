import torch
import torch.nn as nn

from tqdm import tqdm

from torchvision.utils import make_grid
from .base_trainer import BaseTrainer


class CNNTrainer(BaseTrainer):

    def __init__(self, config, log_dir, train_loader, eval_loader=None):
        pass

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
        pass
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