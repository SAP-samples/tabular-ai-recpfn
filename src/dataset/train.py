from torch.utils.data import Dataset
from abc import abstractmethod

from config import Config
from dataset.base import BaseDataset


class TrainDataset(BaseDataset):

    def __init__(self,
                 input_dir,
                 config:Config=None):
        """
        Initialize the dataset and load precomputed data.

        :param input_dir: Root directory containing serialized dataset artifacts
                          (e.g., item sequences, indices, user IDs, embeddings).
        :param config: Optional Config instance controlling dataset and ICL behavior.
        """
        super().__init__(input_dir, config)

    @abstractmethod
    def process_batch(self, inputs):
        """
        Process a batch of inputs based on training configuration.
        :param inputs: Input batch to process.
        :return: Processed batch, excluding last element if not training with ICL.
        """
        outputs = None
        return outputs
