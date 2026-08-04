import torch

from config import Config
from train.base import BaseTrain, logger
from dataset import DefaultTestDataset
from dataset.test import TestDataset


class BaseEvaluate(BaseTrain):
    """
    Base class for evaluation workflows that extend training utilities.
    :return: Provides evaluation-specific methods built on top of BaseTrain.
    """

    def __init__(self,
                 output_dir:str,
                 config:Config=None):
        """
        Initialize the evaluation pipeline with output directory and configuration.
        :param output_dir: Directory to store evaluation outputs.
        :param config: Configuration object for evaluation settings.
        :return: None.
        """
        super().__init__(output_dir, config)

    def load_test_dataset(self, test_dataset:str):
        """
        Load a test dataset based on the provided path.
        :param test_dataset: Path to the test dataset.
        :return: Dataloader for the test dataset.
        """
        dataset = DefaultTestDataset(input_dir=test_dataset, config=self.config)
        test_loader = torch.utils.data.DataLoader(dataset, self.config.batch_size, collate_fn=dataset.process_batch,
                                                  num_workers=0, prefetch_factor=None)
        return test_loader
