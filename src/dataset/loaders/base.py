import numpy as np
import torch
import os
import logging

from config import Config
from dataset.util import get_split


class BaseLoader(object):
    """
    Default dataset loader for clickstream data.
    param input_dir: Directory containing the clickstream data file.
    param config: Configuration object with dataset parameters.
    """
    def __init__(self, input_dir, config:Config):
        """Initialize the loader with input directory and configuration.

        Args:
            input_dir (str): Directory containing the dataset (expects <dir>/<dir_name>.inter).
            config (Config): Configuration with dataset parameters and flags.
        """
        self.input_dir = input_dir
        self.config = config
