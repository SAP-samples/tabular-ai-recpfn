import torch
import random

from sdg.sdg_config import SDGConfig
from sdg.transition_matrix_priors.random_graph import RandomGraph
from sdg.transition_matrix_priors.latent_factor import LatentFactor


class Base(object):

    def __init__(self, num_seqs:int, num_items:int,
                 config:SDGConfig, embedding_store:torch.Tensor=None):

        self.num_seqs = num_seqs
        self.num_items = num_items
        self.config = config
        self.embedding_cache = []
        self.emb_store = embedding_store

        self.get_popularity()
        if self.config.transition_matrix_prior == 'random_graph':
            self.transition_matrix_prior = RandomGraph(num_items, config, embedding_store)
        elif self.config.transition_matrix_prior == 'latent_factor':
            self.transition_matrix_prior = LatentFactor(num_items, config, embedding_store)
        else:
            raise ValueError(f"Unknown transition matrix prior: {self.config.transition_matrix_prior}")

        self.item_transition = self.transition_matrix_prior.item_transition
        self.item_embeddings = self.transition_matrix_prior.item_embeddings

    def generate_embeddings(self, num_embeddings:int) -> torch.Tensor:

        indices = random.sample(range(self.emb_store.shape[0]), num_embeddings)
        embs = self.emb_store[indices,:]
        embs = torch.nn.functional.normalize(embs, dim=-1, p=2)
        self.embedding_cache.append(embs.cpu())

    def get_popularity(self):
        self.popularity = torch.randn(self.num_items+1) * self.config.predefined_popularity_coeffient
        self.popularity = torch.nn.functional.softmax(self.popularity, dim=0)
        self.popularity[0] = 0.

    def get_next_item_probabilities(self, next_item_probs):
        """
        Get next item probabilities based on current items.
        :param next_item_probs: Tensor of next item probabilities (batch_size, num_items)
        :return: Next item probabilities
        """
        pop_item_probs = self.popularity
        next_item_probs = next_item_probs * (1 - self.config.popularity_bias) + pop_item_probs.unsqueeze(0) * self.config.popularity_bias + torch.randn_like(next_item_probs) * self.config.noise_coefficient/(self.num_items+1)
        next_item_probs = torch.clamp(next_item_probs, min=0.0)
        next_item_probs[:,0] = 0.
        return torch.nn.functional.normalize(next_item_probs, dim=-1)

    def postprocess_seqs(self, seqs:torch.Tensor) -> list[list]:
        """
        Postprocess sequences to remove padding and ensure correct shape.
        :param seqs: Sequences tensor of shape (num_seqs, max_seq_len).
        :return: Postprocessed sequences tensor.
        """
        if self.config.variable_seq_len:
            seq_lens = torch.randint(self.config.min_seq_len, self.config.max_seq_len, (seqs.shape[0],))
            seqs = [seq[:seq_len].tolist() for seq, seq_len in zip(seqs, seq_lens)]
        else:
            seqs = seqs.tolist()
            seqs = [seq[:self.config.max_seq_len] for seq in seqs]
        return seqs

    def initialize_users(self):
        """
        Initialize users defined by their starting items
        """
        self.seqs = torch.zeros(self.num_seqs, self.config.max_seq_len, dtype=torch.long)
        self.seqs[:, 0] = torch.randint(1, self.num_items, (self.num_seqs,))

    def step(self, seq_idx:int):
        """
        Step the sequence by transitioning to a new item based on the current item.
        :param seq_idx: Index of the sequence to step
        :return: New item index
        """
        seq_weight = 1
        next_item_probs = 0
        for i in range(max(0, seq_idx-self.config.window_size+1), seq_idx+1):
            current_items = self.seqs[:, i]
            next_item_probs = self.item_transition[current_items, :]
            next_item_probs = self.get_next_item_probabilities(next_item_probs)
            next_item_probs += next_item_probs * seq_weight
            seq_weight *= self.config.decay_coeff
        next_item_probs = torch.nn.functional.normalize(next_item_probs, p=1, dim=-1)
        if not self.config.repeated_items:
            prev_items = self.seqs[:, :seq_idx+1]
            next_item_probs[torch.arange(next_item_probs.size(0)).unsqueeze(-1), prev_items] = 0
        next_item = torch.multinomial(next_item_probs, 1).squeeze(1)
        self.seqs[:, seq_idx+1] = next_item

    def run(self) -> list[list[int]]:
        """
        Run the walk process for all sequences.
        """
        self.initialize_users()
        for seq_idx in range(self.config.max_seq_len - 1):
            self.step(seq_idx)
        self.seqs = self.postprocess_seqs(self.seqs)
        return self.seqs
