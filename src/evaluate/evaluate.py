import torch
import os
import json
from copy import deepcopy
import gc

from config import Config
from evaluate.base import BaseEvaluate, logger
from dataset import DefaultTestDataset
from util.metrics import MetricCollection


class Evaluate(BaseEvaluate):
    """
    Evaluate handles loading a trained model, iterating over test datasets, applying
    temporary configuration overrides per evaluation setup, and collecting metrics.

    :param output_dir: Directory where model and results are stored.
    :param config: Optional Config instance to initialize evaluation settings.
    """

    def __init__(self,
                 output_dir:str,
                 config:Config=None):
        """
        Initialize the Evaluate class.

        :param output_dir: Directory for saving/loading model and results.
        :param config: Optional configuration object for evaluation.
        :return: None
        """
        super().__init__(output_dir, config)

    def update_config(self, config_updates:dict,
                      dataloader:torch.utils.data.DataLoader):
        """
        Apply temporary configuration updates for ICL/evaluation and rebuild the dataloader.

        :param config_updates: Dictionary of config fields to override for this run.
        :param dataloader: Existing test dataloader whose dataset will be reconfigured.
        :return: Tuple of (original_config, updated_dataloader)
        """
        original_config = deepcopy(self.config)
        temp_config = Config(**self.config.to_dict())
        for key, value in config_updates.items():
            setattr(temp_config, key, value)
        self.config = temp_config
        self.model.config = temp_config
        self.baseline_model.config = temp_config
        ds:DefaultTestDataset = dataloader.dataset
        ds.config = temp_config
        ds.icl_dataset.config = temp_config
        dataloader = torch.utils.data.DataLoader(ds, temp_config.batch_size, collate_fn=ds.process_batch,
                                                 num_workers=0, prefetch_factor=None)
        return original_config, dataloader

    def reset_config(self, original_config:Config,
                     dataloader:torch.utils.data.DataLoader):
        """
        Restore the original configuration and rebuild the dataloader accordingly.

        :param original_config: The configuration to restore after temporary updates.
        :param dataloader: The current dataloader whose dataset will be reset to original config.
        :return: Tuple of (original_config, reset_dataloader)
        """
        self.config = original_config
        self.model.config = original_config
        self.baseline_model.config = original_config
        ds:DefaultTestDataset = dataloader.dataset
        ds.config = original_config
        ds.icl_dataset.config = original_config
        dataloader = torch.utils.data.DataLoader(ds, original_config.batch_size, collate_fn=ds.process_batch,
                                                 num_workers=0, prefetch_factor=None)
        return original_config, dataloader

    def run(self, eval_configs:dict = None,
            datasets:list = None,
            save_dir:str = None):
        """
        Execute evaluation over the provided datasets using one or more evaluation configs.
        Loads the trained model, applies per-config overrides, iterates and logs metrics,
        and persists results to disk.

        :param eval_configs: Mapping from config name to dict of config overrides. If None, uses {'default': {}}.
        :param datasets: List of dataset paths to evaluate. If None, uses self.config.test_datasets.
        :param save_dir: Directory to save results.json. Defaults to self.output_dir.
        :return: Dictionary of results indexed by config name and dataset name.
        """
        save_dir = self.output_dir if save_dir is None else save_dir
        model_path = os.path.join(self.output_dir, "model.pth")

        os.makedirs(save_dir, exist_ok=True)
        results_file = f"{save_dir}/results.json"

        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                self.results = json.load(f)
            logger.info(f"Loaded existing results from {results_file}")
        else:
            self.results = {}

        if eval_configs is None:
            eval_configs = {'default': {}}

        for config_name in eval_configs.keys():
            if config_name not in self.results:
                self.results[config_name] = {}

        if datasets is None:
            datasets = self.config.test_datasets

        with torch.no_grad():
            for dataset_path in datasets:
                dataloader = self.load_test_dataset(dataset_path)
                dataset_name = dataset_path.split('/')[-1]
                logger.info(f"Processing dataset: {dataset_name}")

                if not os.path.exists(dataset_path):
                    logger.warning(f"Dataset {dataset_name} not found, skipping...")
                    continue

                if not hasattr(self, 'model') or self.model is None:
                    embedding_dim = dataloader.dataset.emb_dim
                    self.load_model(model_path=model_path, embedding_dim=embedding_dim)
                    self.model.eval()
                    self.baseline_model.eval()

                for config_name, config_updates in eval_configs.items():
                    original_config = deepcopy(self.config)
                    if dataset_name in self.results[config_name]:
                        logger.info(f"Config {config_name} for dataset {dataset_name} already processed, skipping...")
                        continue
                    original_config, dataloader = self.update_config(config_updates, dataloader)
                    logger.info(f"Evaluating with config: {config_name} on dataset: {dataset_name}")

                    metrics, baseline_metrics, _ = self.iterate_one_epoch(
                        model=self.model,
                        baseline_model=self.baseline_model,
                        dataloader=dataloader,
                        optimizer=None,
                        evaluate_last_only=True
                        )

                    with open(results_file, 'w') as f:
                        json.dump(self.results, f, indent=4)
                    logger.info(f"Saved results for config '{config_name}' on dataset '{dataset_name}' to {results_file}")

                    gc.collect()
                    torch.cuda.empty_cache()

                    if dataset_name not in self.results[config_name]:
                        self.results[config_name][dataset_name] = {}
                    self.results[config_name][dataset_name]['trained'] = metrics.to_dict()
                    self.results[config_name][dataset_name]['baseline'] = \
                        baseline_metrics.to_dict() if isinstance(baseline_metrics, MetricCollection) else \
                            {k:v.to_dict() for k,v in baseline_metrics.items()}

                    self.reset_config(original_config, dataloader)

        return self.results
