import torch
import random
from abc import abstractmethod

from sdg.sdg_config import SDGConfig
from constants import DEVICE


class BaseTransitionMatrixPrior(object):

    def __init__(self, num_items:int,
                 config:SDGConfig,
                 embedding_store:torch.Tensor=None):

        self.num_items = num_items
        self.config = config
        self.emb_store = embedding_store
        self.item_embeddings = None
        self.get_item_transition()
        assert self.item_embeddings is not None, "item_embeddings must be initialized in subclass"

    def generate_embeddings(self, num_embeddings:int) -> torch.Tensor:

        indices = random.sample(range(self.emb_store.shape[0]), num_embeddings)
        embs = self.emb_store[indices,:]
        embs = torch.nn.functional.normalize(embs, dim=-1, p=2)
        return embs

    @abstractmethod
    def get_item_transition(self):
        self.item_transition = None
