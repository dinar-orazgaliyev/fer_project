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

        self.logger = logging.getLogger()



    def _configure_logging(self, log_dir):
        
        self.logger.setLevel(1)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s : %(message)s')
        if not os.path.exists(ospj(log_dir)):
            os.mkdir(log_dir)
        _log_file = ospj(log_dir, self.config['name']+".log")
        if os.path.exists(_log_file):
            print(f'Warning! Log file {_log_file} already exists! The logs will be appended!')
        file_handler = logging.FileHandler(_log_file)
        file_handler.setFormatter(formatter)
        if (self.logger.hasHandlers()):
            self.logger.handlers.clear()
        self.logger.addHandler(file_handler)
    


    @abstractmethod # To be implemented by the child classes!
    def _train_epoch(self):
        """
        Training logic for an epoch. Only takes care of doing a single training loop.

        :return: A dict that contains average loss and metric(s) information in this epoch.
        """
        raise NotImplementedError

    def train(self):
        pass

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