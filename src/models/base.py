import torch.nn as nn
from abc import ABC, abstractmethod
from torch.utils.data import Dataset

 # these are passed automatically
 # 'input_channels'
 # 'H'
 # 'W'
 # 'input_size'

class BASE(nn.Module, ABC):
    def __init__(self):
        super(BASE, self).__init__()
        self.train_loader = None
        self.test_loader = None
        self.dataset = None
        self.data_shape = None

    @classmethod
    @abstractmethod
    def supported_dataset(cls):
        """
        Returns a list of supported dataset names for the model.
        Override this method in subclasses to specify supported datasets.
        """
        return Dataset 

    def get_name(self):
        """
        Returns the name of the model class.
        str: The class name
        """
        return self.__class__.__name__



