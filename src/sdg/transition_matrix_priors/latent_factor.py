import torch
import random

from sdg.sdg_config import SDGConfig
from sdg.transition_matrix_priors.base import BaseTransitionMatrixPrior
from constants import DEVICE


class LatentFactor(BaseTransitionMatrixPrior):
    """
    A transition matrix prior that generates random item-to-item transitions.

    This class creates a transition graph where items are randomly connected to other items,
    with optional self-loops based on repeated item probability.
    """

    def __init__(self, num_items:int,
                 config:SDGConfig,
                 embedding_store:torch.Tensor):
        """
        Initialize the LatentFactor class with number of items, configuration, and embedding store.

        :param num_items: Number of items in the system
        :param config: SDGConfig object containing configuration parameters
        :param embedding_store: Precomputed embedding store (optional, can be None)
        """
        super().__init__(num_items, config, embedding_store)

    def initialize_concept_vectors(self):
        self.concept_vectors = self.generate_embeddings(self.config.num_latent_concepts)
        self.item_embeddings = self.item_concepts @ self.concept_vectors
        self.item_embeddings += torch.randn_like(self.item_embeddings) * self.config.noise_coefficient

    def get_concept_matrix(self):
        """
        Compute concept to concept transition matrix

        Note: The transition matrix is asymmetric, meaning that the transition from concept i to concept j
        does not necessarily equal the transition from concept j to concept i.
        """
        self.concept_matrix = torch.rand(self.config.num_latent_concepts, self.config.num_latent_concepts) * 2 -1
        mask = torch.rand(self.config.num_latent_concepts, self.config.num_latent_concepts) <= self.config.concept_matrix_occupancy_rate
        self.concept_matrix *= mask
        mask = torch.rand(self.config.num_latent_concepts) >= self.config.concept_matrix_diagonal_deviation_rate
        self.concept_matrix[mask, mask] = 1.0

    def get_item_concepts(self):
        """
        Randomly assign concepts to items as a (num_items, num_latent_concepts) tensor.
        """
        if self.config.continuous_concept_weights:
            item_concepts = torch.rand(self.num_items+1, self.config.num_latent_concepts) * 2 -1
        else:
            item_concepts = torch.randint(0, 2, (self.num_items+1, self.config.num_latent_concepts)).float() * 2 -1
        mask = torch.rand(self.num_items+1, self.config.num_latent_concepts) <= self.config.mean_num_concepts_per_item / self.config.num_latent_concepts
        self.item_concepts = item_concepts * mask
        self.item_concepts[0,:] = 0.
        self.item_concepts = torch.nn.functional.normalize(self.item_concepts, dim=-1)

    def get_item_transition(self):
        """
        Compute item transition matrix based on item concepts.
        """
        self.get_item_concepts()
        self.get_concept_matrix()
        self.initialize_concept_vectors()
        self.item_transition = self.item_concepts @ self.concept_matrix @ self.item_concepts.T
        self.item_transition.fill_diagonal_(0)
        if self.config.repeated_items:
            mask = torch.rand(self.num_items+1) <= self.config.repeated_item_probability
            self.item_transition[mask, mask] = 1.0
        self.item_transition = torch.nn.functional.softmax(self.item_transition * self.config.logit_scale, dim=-1)
