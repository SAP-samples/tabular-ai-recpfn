import torch
import os
import json

from train.train import Train
from config import Config
from evaluate.evaluate import Evaluate
from constants import SDG_STAGE_1_PARAMS, SDG_STAGE_2_PARAMS

cwd = os.path.dirname(os.path.abspath(__file__))

config = \
    Config(
        train_with_icl=True,
        num_icl_examples=128,
        train_with_synthetic_data=True,
        n_layers=4,
        batch_size=16,
        reg_lambda=0.0001,
        warmup_epochs=6,
        baseline_config = 'balanced',
        learning_rate=1e-4,
        early_stopping_patience=20,
        positional_embedding_scheme='hard-alibi',
        optimizer='adamw',
        llm='qwen1b',
        max_sequence_len=15,
        num_epochs=120,
        use_cached_embeddings=True,
        num_gradient_accumulation_steps=2,
        cached_embedding_folder=os.path.join(cwd, '../embedding_store/beir')
        )

data_dir = os.path.join(cwd, '../datasets/')
test_datasets = \
    [os.path.join(data_dir, dataset) for dataset in os.listdir(data_dir) if (not dataset.startswith('.'))]

eval_configs = {}
eval_configs['base'] = {
        'test_datasets': test_datasets,
        'batch_size': 1,
        'num_icl_examples': 8,
    }


if __name__ == "__main__":
    # Example usage
    output_dir = os.path.join(cwd, f"../output/test_lr_{config.llm}/")

    if False:
        # Training
        if True:
            # Stage 1
            config.sdg_params = SDG_STAGE_1_PARAMS
            trainer = Train(output_dir=output_dir, config=config)
            trainer.run(warm_start=False)

        if True:
            # Stage 2
            config.sdg_params = SDG_STAGE_2_PARAMS
            trainer = Train(output_dir=output_dir, config=config)
            trainer.run(warm_start=True)

    # Evaluation
    evaluator = Evaluate(output_dir=output_dir, config=config)
    results = evaluator.run(eval_configs=eval_configs, datasets=test_datasets)
    print("Evaluation Results:", json.dumps(results, indent=4))
