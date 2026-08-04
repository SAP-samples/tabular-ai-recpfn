import torch
import numpy as np


def combined_dataloader(dataloaders: list[torch.utils.data.DataLoader],
                            restart_on_exhausted: bool = True,
                            max_oversampling_factor: int = 5):
        """
        Combine multiple dataloaders into one.
        Randomly samples batches from dataloaders, weighting larger dataloaders more heavily,
        with a max oversampling factor. Keeps the restart_on_exhausted configuration.

        Args:
            dataloaders (list[torch.utils.data.DataLoader]): List of dataloaders.
            restart_on_exhausted (bool): Whether to restart dataloaders when exhausted.
            max_oversampling_factor (int): Maximum oversampling factor for smaller dataloaders.

        Returns:
            Iterator that yields batches from the dataloaders.
        """
        lengths = [len(dl.dataset) for dl in dataloaders]
        min_length = min(lengths)
        weights = [min(max_oversampling_factor, length/min_length) for length in lengths]
        normalized_weights = [w / sum(weights) for w in weights]

        iterators = [iter(dl) for dl in dataloaders]
        while iterators:
            chosen_index = np.random.choice(len(iterators), p=normalized_weights)
            try:
                yield next(iterators[chosen_index])
            except StopIteration:
                if restart_on_exhausted:
                    iterators[chosen_index] = iter(dataloaders[chosen_index])
                else:
                    del iterators[chosen_index]
                    del normalized_weights[chosen_index]
                    normalized_weights = [w / sum(normalized_weights) for w in normalized_weights]
                    if not iterators:
                        break


def check_nan(inputs:list[torch.Tensor]):
    for i,x in enumerate(inputs):
        if x.numel() > 0:
            if torch.isnan(x).any() or torch.isinf(x).any():
                return f'Input {i} contains NaN or Inf values.'
        else:
            if torch.isnan(x) or torch.isinf(x):
                return f'Input {i} contains NaN or Inf values.'
    return None
