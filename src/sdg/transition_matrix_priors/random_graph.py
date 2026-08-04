import torch
import random

from sdg.sdg_config import SDGConfig
from sdg.transition_matrix_priors.base import BaseTransitionMatrixPrior
from constants import DEVICE


class RandomGraph(BaseTransitionMatrixPrior):
    """
    A transition matrix prior that generates random item-to-item transitions.

    This class creates a transition graph where items are randomly connected to other items,
    with optional self-loops based on repeated item probability.
    """

    def __init__(self, num_items:int,
                 config:SDGConfig,
                 embedding_store:torch.Tensor):
        """
        Initialize the RandomGraph class with number of items, configuration, and embedding store.

        :param num_items: Number of items in the system
        :param config: SDGConfig object containing configuration parameters
        :param embedding_store: Precomputed embedding store (optional, can be None)
        """
        super().__init__(num_items, config, embedding_store)

    def get_item_embeddings(self):
        """
        Generate item embeddings from a heavy-tailed distribution (Student's t-distribution).

        Creates embeddings for all items plus a padding item (index 0), which is set to zero.
        The embeddings are stored in self.item_embeddings.
        """
        self.item_embeddings = self.generate_embeddings(self.num_items+1)
        self.item_embeddings[0,:] = 0.

    def get_item_transition(self):
        """
        Compute item transition matrix based on random item connections.

        Creates a transition matrix where:
        - Each item is randomly connected to mean_num_related_items other items
        - Self-loops are initially disabled (set to -1000)
        - Self-loops are re-enabled for items based on repeated_item_probability
        - Padding item (index 0) has no transitions

        The transition matrix is stored in self.item_transition, and item frequencies
        are initialized in self.item_frequency.
        """
        self.get_item_embeddings()
        self.item_transition = torch.zeros(self.num_items+1, self.num_items+1)
        self.item_transition[0, :] = 0.
        self.item_transition[:, 0] = 0.
        for i in range(self.config.mean_num_related_items):
            self.item_transition[torch.arange(self.num_items).long()+1,
                                torch.randint(1, self.num_items+1, (self.num_items,))] \
                = torch.ones(self.num_items)
        self.item_transition[torch.ones(self.num_items+1).bool(),torch.ones(self.num_items+1).bool()] = -1000.
        mask = torch.rand(self.num_items+1) <= self.config.repeated_item_probability
        self.item_transition[mask, mask] = 1.0
