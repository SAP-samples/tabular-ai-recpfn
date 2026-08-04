import torch
import random

from config import Config
from dataset.base import BaseDataset
from dataset.train import TrainDataset
from sdg.generate import Generate


class SyntheticDataset(TrainDataset, BaseDataset):
    """
    A synthetic dataset that generates training data on-the-fly for recommendation models.
    Inherits from TrainDataset and BaseDataset to provide synthetic sequences and embeddings.
    """

    def __init__(self, config:Config=None):
        """
        Initialize the SyntheticDataset with a configuration object.
        :param config: Configuration object containing dataset generation parameters including
                      batch size, number of ICL examples, and SDG item count.
        """
        self.config = config
        self.gen = Generate(config)
        self.emb_dim = self.gen.emb_dim

    def __getitem__(self, index:int) -> int:
        """
        Get a single dataset item by index.
        :param index: Index of the item to retrieve.
        :return: The index value itself (used as a placeholder for batch generation).
        """
        return index

    def process_batch(self, indices:list):
        """
        Process a batch of indices to generate synthetic sequences, embeddings, and labels.
        :param indices: List of indices representing the batch items to process.
        :return: Tuple of (input_embs, mask, labels, embs, icl_embs, config) if train_with_icl is True,
                otherwise returns a subset excluding the last element.
        """
        batch_size = len(indices)
        num_seqs = batch_size + self.config.num_icl_examples

        seqs, embs, config = self.gen.get_one_dataset(num_seqs=num_seqs,
                                                      num_items=self.config.num_sdg_items)

        icl = seqs[self.config.batch_size:]
        seqs = seqs[:self.config.batch_size]

        input_seqs, mask, labels = self.process_item_seq(seqs)
        input_embs = embs[input_seqs,:]

        max_icl_len = max(len(seq) for seq in icl)
        icl = torch.Tensor([seq + [0]*(max_icl_len - len(seq)) for seq in icl]).long()
        icl_embs = embs[icl,:]

        inputs = (input_embs, mask, labels, embs, icl_embs, config)

        if self.config.train_with_icl:
            return inputs
        else:
            return inputs[:-1]

    def __len__(self):
        """
        Get the length of the dataset.
        :return: Fixed value of 10000 representing the number of synthetic samples available per epoch.
        """
        return 10000
