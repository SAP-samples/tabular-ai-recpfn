import torch
import random
import logging

from config import Config
from constants import DEVICE
from context_selector.selector import InvertedIndexMatcher
from dataset.base import BaseDataset
from dataset.train import TrainDataset
from dataset.test import TestDataset
from dataset.loaders.default import DefaultLoader


class DefaultDataset(BaseDataset):

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
        self.load_data()

    def load_data(self):
        """
        Load dataset artifacts into memory using DefaultLoader and attach them to the instance.

        Side effects:
        - Populates the following attributes:
          - self.items_ls: List[List[int]] of user item sequences.
          - self.idx_dict (dict): Mapping from split name to list of indices into all_seqs.
          - self.uids: List of user IDs.
          - self.item_ids: List of global item IDs aligned with the embeddings.
          - self.embeddings: torch.Tensor or np.ndarray of shape [N, D] with item embeddings.

        :return: None
        """
        self.data = DefaultLoader(self.input_dir, self.config)
        self.items_ls, self.idx_dict, self.uids, self.item_ids, self.embeddings = \
            self.data.items_ls, self.data.idx_dict, self.data.uids, self.data.item_ids, self.data.embeddings
        self.emb_dim = self.embeddings.shape[1]


class DefaultTrainDataset(TrainDataset, DefaultDataset):
    """
    Default training dataset class combining TrainDataset and DefaultDataset functionalities.
    """

    def __init__(self,
                 input_dir,
                 config:Config=None,
                 mode:str='train'):
        """
        Initialize the DefaultTrainDataset with a directory and configuration.

        :param input_dir: Root directory containing serialized dataset artifacts.
        :param config: Optional Config instance controlling dataset and ICL behavior.
        :param mode: One of {'train', 'validate'} indicating the dataset mode.
        """
        super().__init__(input_dir, config)
        self.mode = mode

    def get_one_batch(self, indices:list=None) -> tuple[list[list[int]], torch.Tensor, dict]:
        sampled_seqs = [self.items_ls[i] for i in indices]
        embs = self.embeddings
        config = self.config.to_dict()
        return sampled_seqs, embs, config

    def __getitem__(self, index:int) -> tuple[torch.Tensor]:
        return index

    def process_batch(self, indices:list):
        seqs, embs, config = self.get_one_batch(indices)
        input_seqs, mask, labels = self.process_item_seq(seqs)

        icl_indices = random.sample(self.idx_dict['train'], self.config.num_icl_examples + self.config.batch_size)
        icl_indices = list(set(icl_indices) - set(indices))
        icl_seqs = [self.items_ls[i] for i in icl_indices[:self.config.num_icl_examples]]

        input_embs = embs[input_seqs,:]
        output, labels = torch.unique(labels.view(-1), return_inverse=True)
        sample_embs = embs[output,:]
        labels = labels.view(input_seqs.shape[0], input_seqs.shape[1])

        icl = [seq[-self.config.max_sequence_len:] for seq in icl_seqs]
        max_icl_len = max(len(seq) for seq in icl)
        icl = torch.Tensor([seq + [0]*(max_icl_len - len(seq)) for seq in icl]).long()
        icl_embs = embs[icl,:]

        inputs = (input_embs, mask, labels, sample_embs, icl_embs, config)

        if self.config.train_with_icl:
            return inputs
        else:
            return inputs[:-1]

    def __len__(self):
        return 10000


class DefaultTestDataset(TestDataset, DefaultDataset):
    """
    Default testing dataset class combining TestDataset and DefaultDataset functionalities.
    """

    def __init__(self,
                    input_dir,
                    config:Config=None):
        """
        Initialize the DefaultTestDataset with a directory and configuration.

        :param input_dir: Root directory containing serialized dataset artifacts.
        :param config: Optional Config instance controlling dataset and ICL behavior.
        """
        super().__init__(input_dir, config)
        self.embeddings = torch.Tensor(self.embeddings).to(DEVICE)
        train_seqs = [self.items_ls[i] for i in self.idx_dict['train']]
        train_uids = [self.uids[i] for i in self.idx_dict['train']]
        self.set_icl_dataset(train_seqs, train_uids)

    def __len__(self):
        """
        Number of sequences in the dataset.

        :return: Integer dataset size.
        """
        return len(self.idx_dict['test'])

    def __getitem__(self, index):
        return self.data.__getitem__(index, split='test')
