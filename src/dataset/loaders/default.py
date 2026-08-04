import numpy as np
import torch
import os
import logging
import pandas as pd
import json

from config import Config
from dataset.util import get_split
from dataset.loaders.base import BaseLoader


class DefaultLoader(BaseLoader):
    """
    Default dataset loader for clickstream data.
    param input_dir: Directory containing the clickstream data file.
    param config: Configuration object with dataset parameters.
    """

    CLICKSTREAM_FILENAME = 'clickstream.csv'
    EMBEDDINGS_FILENAME = 'item_embeddings.npy'
    ITEM_IDS_FILENAME = f"item_ids.json"

    def __init__(self, input_dir, config:Config):
        """Initialize the loader with input directory and configuration.

        Args:
            input_dir (str): Directory containing the dataset (expects <dir>/<dir_name>.inter).
            config (Config): Configuration with dataset parameters and flags.
        """
        super().__init__(input_dir, config)
        self.items_ls, self.idx_dict, self.uids, self.item_ids, self.embeddings = self.process()

    def __getitem__(self, index, split=None):
        """Get item sequences for a given index or split.

        Args:
            index (int): Index of the sequence to retrieve.
            split (str, optional): If provided, retrieves from the specified split ('train', 'valid', 'test').

        Returns:
            list[str] | list[list[str]]: ItemId sequence(s) for the specified index or split.
        """
        if split is not None:
            indices = self.idx_dict[split]
            return self.items_ls[indices[index]]
        return self.items_ls[index]

    def process(self):
        """Process the dataset by loading clickstream, item IDs, and embeddings.

        Loads raw clickstream data, processes it into sequences and splits,
        loads item IDs and passage embeddings.

        Returns:
            tuple:
                - all_seqs (list[list[str]]): List of itemId sequences for all users.
                - idx_dict (dict): Mapping from split name to list of indices into all_seqs.
                - uids (list[str]): UserIds aligned with all_seqs order.
                - item_ids (dict | list): Item IDs or mapping depending on file contents.
                - embeddings (torch.Tensor): Passage/item embeddings tensor.
        """
        clickstream = self.load_clickstream()
        all_seqs, idx_dict, uids = self.process_clickstream(clickstream)
        item_ids = self.load_item_ids()
        embeddings = self.load_embeddings(f"{self.config.llm}_{self.EMBEDDINGS_FILENAME}")
        id_to_idx = {id:idx for idx,id in enumerate(item_ids)}
        all_seqs = [[id_to_idx.get(itm, 0) for itm in ls] for ls in all_seqs]
        return all_seqs, idx_dict, uids, item_ids, embeddings

    def load_clickstream(self):
        """Load the raw clickstream interactions from the .inter file.

        Reads <input_dir>/<dir_name>.inter with columns user_id:token, item_id:token, timestamp:float,
        renames them to userId, itemId, timestamp, and optionally truncates to max_clickstream_events.

        Returns:
            pd.DataFrame: DataFrame with columns ['userId', 'itemId', 'timestamp'].
        """
        cs_filepath = os.path.join(self.input_dir, f'{self.input_dir.split("/")[-1]}.inter')
        if self.config.verbose:
            logging.info(f'Loading from {cs_filepath}')
        usecols = ['user_id:token', 'item_id:token', 'timestamp:float']
        clickstream = pd.read_csv(
            cs_filepath,
            delimiter='\t',
            usecols=usecols,
            dtype={'user_id:token': str, 'item_id:token': str, 'timestamp:float': float},
        )
        clickstream.rename(columns={'user_id:token':'userId', 'item_id:token':'itemId', 'timestamp:float':'timestamp'}, inplace=True)
        if self.config.max_clickstream_events:
            clickstream = clickstream.head(self.config.max_clickstream_events)
        return clickstream

    def process_clickstream(self, cs:pd.DataFrame):
        """Convert raw clickstream into per-user sequences and split indices.

        Uses or creates <input_dir>/<dir_name>_split.json to define user splits, sorts by userId and timestamp,
        aggregates itemId sequences per user, drops sequences of length 1, and builds indices per split.

        Args:
            cs (pd.DataFrame): Clickstream with columns ['userId', 'itemId', 'timestamp'].

        Returns:
            tuple[list[list[str]], dict[str, list[int]], list[str]]:
                - all_seqs: concatenated list of itemId sequences for all users across splits.
                - idx_dict: mapping from split name to list of indices into all_seqs.
                - uids: userIds aligned with all_seqs order.
        """
        if os.path.exists(os.path.join(self.input_dir, f'{self.input_dir.split("/")[-1]}_split.json')):
            with open(os.path.join(self.input_dir, f'{self.input_dir.split("/")[-1]}_split.json'), 'r') as f:
                split_dict = json.load(f)
        else:
            user_ids = cs['userId'].unique()
            split_dict = get_split(user_ids.tolist(),
                                   split_ratio=self.config.data_split_ratio,
                                   seed=self.config.random_seed,
                                   max_test_size=self.config.max_test_users)
            with open(os.path.join(self.input_dir, f'{self.input_dir.split("/")[-1]}_split.json'), 'w') as f:
                json.dump(split_dict, f)

        idx_dict = {}
        uids = []
        all_seqs = []
        cs.sort_values(by=['userId', 'timestamp'], inplace=True)

        for split in split_dict.keys():
            user_ids = set(split_dict[split])
            tmp_cs = cs[cs['userId'].isin(user_ids)].copy()
            grouped = tmp_cs.groupby('userId', sort=False)['itemId'].apply(list).reset_index()
            seqs = grouped['itemId'].tolist()
            seqs = [ls for ls in seqs if len(ls)!=1]
            all_seqs.extend(seqs)
            uids.extend(grouped['userId'].tolist())
            idx_dict[split] = list(range(len(all_seqs)-len(seqs), len(all_seqs)))

        return all_seqs, idx_dict, uids

    def load_item_ids(self):
        """Load item identifier mapping for the current LLM variant.

        Builds the path as <input_dir>/<llm>_item_ids.json and reads JSON.

        Returns:
            dict | list: Item IDs or mapping depending on file contents.
        """
        iid_filepath = os.path.join(self.input_dir, f"{self.config.llm}_{self.ITEM_IDS_FILENAME}")
        if self.config.verbose:
            logging.info(f'Loading from {iid_filepath}')
        with open(iid_filepath, 'rb') as f:
            item_ids = json.load(f)
        return item_ids

    def load_embeddings(self, embedding_filename):
        """Load passage/item embeddings from a NumPy .npy file.

        Args:
            embedding_filename (str): Filename inside input_dir (e.g., 'passage_embeddings.npy').

        Returns:
            torch.Tensor: Embeddings tensor; normalized if config.normalize_embs is True.
        """
        emb_filepath = os.path.join(self.input_dir, embedding_filename)
        print(f'Loading embeddings from {emb_filepath}')
        if self.config.verbose:
            logging.info(f'Loading from {emb_filepath}')
        with open(emb_filepath, 'rb') as f:
            emb = torch.Tensor(np.load(f))
        if self.config.normalize_embs:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb
