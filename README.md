# RecPFN: Prior-Fitted Networks for In-Context-Based Recommendations

[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/tabular-ai-recpfn)](https://api.reuse.software/info/github.com/SAP-samples/tabular-ai-recpfn)

## Description

RecPFN is a lightweight in-context learning (ICL) recommender system that performs single-pass next-item prediction using a small support set. It adapts to new domains at inference time without per-domain fine-tuning, using a transformer architecture trained on synthetic sequential interaction data.

Key features:
- Sequential recommendation with in-context adaptation (no per-domain fine-tuning)
- Two-stage synthetic data generation (SDG) using random graph and latent factor transition matrix priors
- Hard-ALiBi positional encoding with alternating self-attention and cross-attention layers
- Dataset format follows [RecBole](https://recbole.io/) schema

## Requirements

- Python 3.9+
- PyTorch (GPU recommended)
- See `requirements.txt` for the complete environment

```bash
pip install -r requirements.txt
```

## Download and Installation

```bash
git clone https://github.com/SAP-samples/tabular-ai-recpfn.git
cd tabular-ai-recpfn
pip install -r requirements.txt
```

### Data

Datasets must follow RecBole's schema. Place each dataset under `datasets/DATASET_NAME/` containing at minimum:
- `DATASET_NAME.inter` — tab-separated interactions with columns `user_id:token`, `item_id:token`, `timestamp:float`
- `DATASET_NAME.item` — item metadata
- `<LLM_NAME>_item_embeddings.npy` — NumPy array of shape `[num_items, emb_dim]`
- `<LLM_NAME>_item_ids.json` — ordered list of item IDs corresponding to embedding rows

### Embedding Store

Training requires a global embedding store for synthetic data generation:

```
embedding_store/
  beir/
    <LLM_NAME>.pt   # torch.Tensor of shape [N, emb_dim]
```

Embeddings used in the paper will be provided at a later stage.

### Quickstart

Edit `src/main.py` to set your dataset paths and `llm` name, then run:

```bash
python src/main.py
```

Training is controlled by the `if False:` block in `main.py` — set to `True` to enable. Two training stages are supported: Stage 1 uses `random_graph` priors only; Stage 2 adds `latent_factor` priors. Evaluation runs automatically after training.

### Configuration

All hyperparameters are in `src/config.py` (`Config` class). Override via keyword arguments when constructing `Config` in `src/main.py`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `llm` | `'baai'` | Embedding source label (must match filename prefix) |
| `n_layers` | `2` | Number of transformer layers |
| `max_sequence_len` | `100` | Maximum input sequence length |
| `train_with_synthetic_data` | `False` | Use SDG instead of real datasets for training |
| `train_with_icl` | `False` | Enable in-context learning during training |
| `num_icl_examples` | `128` | Number of ICL support sequences |
| `positional_embedding_scheme` | `'hard-alibi'` | One of `alibi`, `hard-alibi`, `abs`, `nope` |

### Outputs

Checkpoints (`model.pth`) and evaluation results (`results.json`) are saved to the configured `output_dir`.

## Known Issues

No known issues.

## How to obtain support

[Create an issue](https://github.com/SAP-samples/tabular-ai-recpfn/issues) in this repository if you find a bug or have questions about the content.

For additional support, [ask a question in SAP Community](https://answers.sap.com/questions/ask.html).

## Contributing

If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License

Copyright 2026 SAP SE or an SAP affiliate company and tabular-ai-recpfn contributors. Please see our [LICENSE](LICENSE) for copyright and license information. Detailed information including third-party components and their licensing/copyright information is available [via the REUSE tool](https://api.reuse.software/info/github.com/SAP-samples/tabular-ai-recpfn).
