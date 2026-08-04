import numpy as np
from collections import Counter, defaultdict
import heapq
from typing import List, Tuple, Dict
import random

import time
from itertools import chain
import logging

from config import Config


class InvertedIndexMatcher(object):
    """
    Inverted index-based sequence matcher for efficient candidate retrieval.

    Uses an inverted index to quickly find sequences that share items with a query sequence,
    then ranks candidates by Jaccard similarity (with optional weighted variant).
    """

    def __init__(self, sequences: List[List[int]],
                 config: Config = None, user_ids: List[int] = None):
        """
        Initialize the InvertedIndexMatcher.

        :param sequences: List of sequences where each sequence is a list of item IDs.
        :param config: Configuration object containing matcher parameters.
        :param user_ids: Optional list of user IDs corresponding to each sequence.
        """
        self.config = config
        self.sequences = sequences
        self.user_ids = user_ids
        self.preprocess_sequences()
        self.build_index()
        self._seq_sets = [set(seq[:-1]) if seq else set() for seq in self.sequences]
        self._seq_lens = np.array([len(s) for s in self._seq_sets], dtype=np.int32)

    def preprocess_sequences(self):
        """
        Preprocess sequences by applying maximum sequence length constraint.

        Splits longer sequences into overlapping chunks using a sliding window approach
        to respect the configured maximum sequence length.
        """
        max_seq_len = self.config.max_sequence_len
        if max_seq_len is None:
            return

        step = max(1, int(max_seq_len / 2))
        new_sequences = []
        new_uids = []
        for uid, seq in zip(self.user_ids, self.sequences):
            if len(seq) <= max_seq_len:
                new_sequences.append(seq)
                new_uids.append(uid)
            else:
                new_sequences.extend(seq[start:start + max_seq_len]
                                    for start in range(0, len(seq), step))
                new_uids.extend([uid] * ((len(seq) - 1) // step + 1))
        self.sequences = new_sequences
        self.user_ids = new_uids

    def build_index(self):
        """
        Build inverted index mapping items to sequence indices.

        Creates a dictionary where each item ID maps to the set of sequence indices
        that contain that item. Excludes the last item in each sequence as it represents
        the prediction target.
        """
        self.inverted_index = defaultdict(set)

        for seq_idx, sequence in enumerate(self.sequences):
            for item in set(sequence[:-1]):
                self.inverted_index[item].add(seq_idx)

    def find_closest_batch(self, query: List[int], k: int = 2, total_k: int = 1) -> List[Tuple[int, float]]:
        """
        Find the closest sequences to the query using recency-weighted retrieval.

        Implements a layered candidate retrieval strategy that prioritizes recent items
        in the query sequence(s). Supports both weighted and unweighted Jaccard similarity.

        :param query: Query sequence (list of item IDs) or batch of sequences (list of lists).
        :param k: Number of recent items to consider for candidate retrieval (recency layers).
        :param total_k: Number of top similar sequences to return.
        :return: List of tuples containing (sequence, similarity_score, user_id) for the top-k
                 most similar sequences, sorted by descending similarity.
        """
        max_candidates = int(getattr(self.config, "max_candidates_cap", total_k * 10))
        if not query:
            return []

        if isinstance(query[0], list):
            priority_groups = []
            for depth in range(k):
                group = []
                for seq in query:
                    if len(seq) > depth:
                        group.append(seq[-1 - depth])
                if not group:
                    break
                priority_groups.append(group)
        else:
            priority_groups = []
            for depth in range(min(k, len(query))):
                priority_groups.append([query[-1 - depth]])

        candidate_counter = Counter()
        for i, group in enumerate(priority_groups):
            for item in set(group):
                cand = self.inverted_index.get(item)
                if cand:
                    candidate_counter.update(cand)
            if len(candidate_counter) >= max_candidates:
                break
            if i == 1 and not candidate_counter:
                break
        if not candidate_counter:
            return []

        candidate_indices = [idx for idx, _ in candidate_counter.most_common(max_candidates)]
        query_items = set(chain(*priority_groups))

        use_weighted = bool(getattr(self.config, "weighted_icl_sampling", False))
        decay_rate = float(getattr(self.config, "weighted_decay_rate", 0.5))
        batch_query = isinstance(query[0], list) if query else False

        if use_weighted:
            if batch_query:
                weights_dict = {}
                for depth, group in enumerate(priority_groups):
                    w = float(np.exp(-decay_rate * depth))
                    for item in group:
                        if w > weights_dict.get(item, 0.0):
                            weights_dict[item] = w
                weighted_items = list(weights_dict.keys())
                weights_container = weights_dict
            else:
                weighted_items = list(query)
                positions = np.arange(len(query), dtype=np.float32)
                weights_list = np.exp(-decay_rate * (len(query) - 1 - positions)).tolist()
                weights_container = weights_list

        top_k = total_k
        heap: List[Tuple[float, int]] = []
        push = heapq.heappush
        pop = heapq.heappop

        for idx in candidate_indices:
            seq_set = self._seq_sets[idx]
            if not seq_set:
                continue
            if use_weighted:
                if isinstance(weights_container, dict):
                    num = 0.0
                    den = 0.0
                    all_items = seq_set | set(weighted_items)
                    for item in all_items:
                        w1 = weights_container.get(item, 0.0)
                        w2 = 1.0 if item in seq_set else 0.0
                        num += min(w1, w2)
                        den += max(w1, w2)
                    sim = (num / den) if den > 0.0 else 0.0
                else:
                    wdict = {itm: w for itm, w in zip(weighted_items, weights_container)}
                    num = 0.0
                    den = 0.0
                    all_items = seq_set | set(weighted_items)
                    for item in all_items:
                        w1 = wdict.get(item, 0.0)
                        w2 = 1.0 if item in seq_set else 0.0
                        num += min(w1, w2)
                        den += max(w1, w2)
                    sim = (num / den) if den > 0.0 else 0.0
            else:
                inter = len(seq_set & query_items)
                if inter == 0:
                    continue
                union = len(seq_set | query_items)
                sim = inter / union if union > 0 else 0.0

            if len(heap) < top_k:
                push(heap, (sim, idx))
            else:
                if sim > heap[0][0]:
                    pop(heap)
                    push(heap, (sim, idx))

        if not heap:
            return []

        heap.sort(key=lambda x: x[0], reverse=True)
        similarities = [(self.sequences[idx], float(sim),
                         self.user_ids[idx] if self.user_ids is not None else None) for sim, idx in heap]
        return similarities
