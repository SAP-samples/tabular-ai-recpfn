from torch.utils.data import Dataset
import torch

from config import Config


class BaseDataset(Dataset):

    def __init__(self,
                 input_dir,
                 config:Config=None):
        """
        Initialize the Preprocess class with a directory and configuration.
        """
        self.config = config
        self.input_dir = input_dir

    def process_item_seq(self, seqs, return_label=True):
        seqs = [seq[-self.config.max_sequence_len:] for seq in seqs]
        batch_max_len = max(list(map(len, seqs)))
        input_seqs = torch.Tensor([seq + [0]*(batch_max_len - len(seq)) for seq in seqs]).long()
        if not return_label:
            return input_seqs, input_seqs!=0
        labels = input_seqs[:, 1:].clone()
        mask = labels!=0
        input_seqs = (input_seqs[:, :-1].clone()) * mask.long()
        return input_seqs, mask, labels
