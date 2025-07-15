import logging
import os
from abc import abstractmethod

import torch
from numpy import inf
from os.path import join as ospj


try:
    import wandb
except:
    pass
class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, config, log_dir):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.epochs = config['train_args']['epochs']
        self._device = torch.device(config['train_args']['device'])
        #self._configure_logging(log_dir=log_dir)



    


    @abstractmethod # To be implemented by the child classes!
    def _train_epoch(self):
        """
        Training logic for an epoch. Only takes care of doing a single training loop.

        :return: A dict that contains average loss and metric(s) information in this epoch.
        """
        raise NotImplementedError

    def train(self):
        self.logger.info("----New Training Session----")

        for epoch in range(self.epochs + 1):
            self.logger.info(f"epoch {epoch}")
            self.current_epoch = epoch
            train_results = self._train_epoch()
            log = {"epoch":epoch}
            log.update(train_results)

    def should_evaluate(self):
        """
        Based on the self.current_epoch and self.eval_interval, determine if we should evaluate.
        You can take hint from saving logic implemented in BaseTrainer.train() method

        returns a Boolean
        """
        ###  TODO  ################################################
        # Based on the self.current_epoch and self.eval_interval, determine if we should evaluate.
        # You can take hint from saving logic implemented in BaseTrainer.train() method
        # eval_period??
        return self.current_epoch % self.eval_period == 0
        #########################################################
    
    @abstractmethod # To be implemented by the child classes!
    def evaluate(self, loader=None):
        """
        Evaluate the model on the val_loader given at initialization

        :param loader: A Dataloader to be used for evaluation. If not given, it will use the 
        self._eval_loader that's set during initialization..
        :return: A dict that contains metric(s) information for validation set
        """
        raise NotImplementedError
    
    def save_model(self, path=None):
        """
        Saves only the model parameters.
        : param path: path to save model (including filename.)
        """
        self.logger.info("Saving checkpoint: {} ...".format(path))
        torch.save(self.model.state_dict(), path)
        self.logger.info("Checkpoint saved.")
    
    def load_model(self, path=None):
        """
        Loads model params from the given path.
        : param path: path to save model (including filename.)
        """
        self.logger.info("Loading checkpoint: {} ...".format(path))
        self.model.load_state_dict(torch.load(path))
        self.logger.info("Checkpoint loaded.")