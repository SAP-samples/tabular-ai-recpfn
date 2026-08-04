import numpy as np


def get_split(user_ids:list, split_ratio:tuple=(0.8, 0.1, 0.1), seed:int=42,
              max_test_size:int=None):
    """
    Splits user IDs into train, validation, and test sets based on the provided split ratio.
    param user_ids: List of user IDs to be split.
    param split_ratio: Tuple indicating the ratio for train, validation, and test splits.
    param seed: Random seed for reproducibility.
    param max_test_size: Maximum size for the test set. If None, no limit is applied.
    return: A dictionary with keys 'train', 'valid', and 'test' mapping to lists of user IDs.
    """
    np.random.seed(seed)
    np.random.shuffle(user_ids)
    num_users = len(user_ids)
    test_size = min(int(num_users * split_ratio[2]), max_test_size) if max_test_size else int(num_users * split_ratio[2])
    val_size = min(int(num_users * split_ratio[1]), max_test_size) if max_test_size else int(num_users * split_ratio[1])
    train_end = num_users - (test_size + val_size)
    split_dict = {
        'train': user_ids[:train_end],
        'valid': user_ids[train_end:train_end + val_size],
        'test': user_ids[train_end + val_size:train_end + val_size + test_size]
    }
    return split_dict
