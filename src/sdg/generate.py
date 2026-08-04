import os
import random
import torch
import json

from config import Config
from sdg.sdg_config import SDGConfig
from sdg.base import Base as BaseSDG


class Generate:

    def __init__(self, config:Config):
        self.config = config
        self.init_embedding_store(config)

        sdg_config = self.sample_dataset_params()
        self.debug_dataset = BaseSDG(100, 1000, sdg_config, embedding_store=self.emb_store).run()

    def init_embedding_store(self, config:Config):
        print(f"Using cached LLM embeddings from folder: {config.cached_embedding_folder}, model: {config.llm}")
        self.emb_store = torch.load(os.path.join(config.cached_embedding_folder, f'{config.llm}.pt'))
        self.emb_dim = self.emb_store.shape[-1]

    def get_one_dataset(self, num_seqs, num_items) -> tuple[torch.Tensor, list[list[int]]]:
        sdg_config = self.sample_dataset_params()
        ds = BaseSDG(
            num_seqs=num_seqs,
            num_items=num_items,
            config=sdg_config,
            embedding_store=self.emb_store
        )
        seqs = ds.run()
        embs = ds.item_embeddings
        return seqs, embs, sdg_config

    def sample_dataset_params(self) -> SDGConfig:
        """
        Sample a configuration for the generation process.
        """
        if self.config.sdg_params is None:
            raise ValueError("SDG parameters not specified in the configuration.")

        vars = \
            {k: ( random.choice(v) if isinstance(v, list)
              else ( random.randint(int(v['min']), int(v['max'])) if v.get('type', 'float')=='int'
                    else random.uniform(float(v['min']), float(v['max'])) ) ) for k, v in self.config.sdg_params.items()}
        config = SDGConfig(**vars)
        return config

    def save_dataset(self, seqs:list[list[int]],
                     embs:torch.Tensor,
                     config:SDGConfig,
                     output_dir, dataset_idx):
        """
        Save the generated dataset to the specified directory.
        """
        if not os.path.exists(os.path.join(output_dir, f'dataset_{dataset_idx}')):
            os.makedirs(os.path.join(output_dir, f'dataset_{dataset_idx}'))

        save_dir = os.path.join(output_dir, f'dataset_{dataset_idx}')
        with open(os.path.join(save_dir, 'sequences.json'), 'w') as f:
            json.dump(seqs, f)
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config.to_dict(), f, indent=4)
        torch.save(embs, os.path.join(save_dir, 'embeddings.pt'))

    def generate(self):
        """
        Generate datasets based on the configuration.
        """
        for i in range(self.config.num_datasets):
            seqs, embs, config = self.get_one_dataset()
            self.save_dataset(seqs, embs, config, self.output_dir, i)
            print(f"Dataset {i+1}/{self.config.num_datasets} generated and saved.")

        print("All datasets generated successfully.")
