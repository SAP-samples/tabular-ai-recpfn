import torch
from torch.utils.data import Dataset

from config import Config
from constants import DEVICE
from context_selector.selector import InvertedIndexMatcher
from dataset.base import BaseDataset


class TestDataset(BaseDataset):

    embeddings:torch.Tensor

    def __init__(self,
                 input_dir,
                 config:Config=None):
        """
        Initialize the test dataset and load precomputed data.

        :param input_dir: Root directory containing serialized dataset artifacts
                          (e.g., item sequences, indices, user IDs, embeddings).
        :param config: Optional Config instance controlling dataset and ICL behavior.
        """
        super().__init__(input_dir, config)

    def set_icl_dataset(self, seqs:list[list], user_ids:list=None):
        """
        Build the in-context learning (ICL) matcher over a subset of user sequences.

        :param seqs: List of historical item ID sequences (each element is a List[int]).
        :param user_ids: Optional list of user IDs aligned with `seqs`. If provided,
                         len(user_ids) must equal len(seqs).
        :return: None
        """
        self.icl_dataset = InvertedIndexMatcher(sequences=seqs, config=self.config, user_ids=user_ids)

    def get_icl(self, query_seqs:list[list]=None):
        """
        Retrieve ICL support examples for the provided query sequences.

        :param query_seqs: List of sequences (List[List[int]]) whose neighbors should be retrieved.
                           If None, the matcher may decide internally based on its configuration.
        :return: Tuple (icl_embs, icl_mask, icl_user_ids), where:
                 - icl_embs: torch.Tensor of shape [B, L, D] with embeddings for the retrieved ICL sequences,
                             or None if no sequences are found.
                 - icl_mask: torch.Tensor of shape [B, L] with 1/0 padding mask, or None.
                 - icl_user_ids: List of user IDs corresponding to retrieved sequences, or None.
        """
        out = self.icl_dataset.find_closest_batch(query_seqs, total_k=self.config.num_icl_examples,
                                                  k=self.config.icl_k)
        seqs = [x[0] for x in out]
        user_ids = [x[2] for x in out]
        if seqs:
            input_seqs, mask = self.process_item_seq(seqs, return_label=False)
            input_embs = self.embeddings[input_seqs,:]
            return input_embs, mask, user_ids
        else:
            return None, None, None

    def process_batch(self, seqs):
        """
        Vectorize a batch of sequences and prepare embeddings, masks, and labels for evaluation.

        :param seqs: List[List[int]] item ID sequences.
        :return: List with the following elements:
                 [input_embs, mask, labels, all_embeddings, (optional icl_embs), (optional icl_user_ids), meta]
                 - input_embs: torch.Tensor [B, L, D] input embeddings for each position.
                 - mask: torch.Tensor [B, L] 1/0 mask where 1 denotes a valid (non-padding) position.
                 - labels: torch.Tensor [B, L] target item indices over the full item vocabulary.
                 - all_embeddings: torch.Tensor [N, D] full embedding matrix for all items.
                 If config.train_with_icl is True, the list includes:
                 - icl_embs: torch.Tensor [B, K, D] ICL example embeddings, or None if not available.
                 - icl_uids: List of user IDs corresponding to retrieved ICL sequences, or None.
                 - meta: dict reserved for auxiliary information.
        """
        input_seqs_list = [seq[:-1] for seq in seqs]
        input_seqs, mask, labels = self.process_item_seq(seqs)
        input_embs = self.embeddings[input_seqs,:]

        outputs = [input_embs.detach().to(DEVICE), mask, labels, self.embeddings]
        if self.config.train_with_icl:
            icl_embs, icl_mask, icl_uids = self.get_icl(input_seqs_list)
            outputs.append(icl_embs.detach() if icl_embs is not None else None)
        outputs.append(dict())
        return outputs
