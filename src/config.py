import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:

    drop_cold_start_items = True

    max_clickstream_events = None

    learning_rate = 1e-3
    batch_size = 128
    eval_batch_size = 1
    random_seed = 42
    softmax_sample_size = 4000
    val_k = 10
    train_epoch_size = 500 # num batches
    val_epoch_size = 200 # num batches
    num_epochs = 100
    warmup_epochs = 0
    normalize_embs = True
    early_stopping_patience = 8
    num_gradient_accumulation_steps = 1
    loss = 'sce' # Implemented: sce, mse
    reg_lambda = 0.0 # Regularization coefficient for L2 norm of model parameters
    optimizer = 'adam' # Implemented: adam, adamw
    early_stopping_metric = 'hr@10' # Implemented: 'hr@k', 'mrr@k', k should be in eval_k
    log_time = False

    # dataset configs
    train_with_synthetic_data = False
    max_dataset_oversampling = 10
    max_test_users = 10000
    train_datasets = []
    test_datasets = []
    llm = 'baai'
    embedding_dim = 128
    sdg_params = None
    num_sdg_items = 1000
    use_cached_embeddings = False
    cached_embedding_folder = os.path.join(CURRENT_DIR, '..', 'emb_store', 'beir')
    data_split_ratio = (0.8, 0.1, 0.1)

    # backbone configuration
    n_heads = 8
    dropout = 0.2
    n_layers = 2
    baseline_config = 'balanced' # Implemented: 'balanced'
    max_sequence_len = 100
    project_v = True # Whether to project the value embeddings in attention layers
    icl_module_type = 'alternating' # Implemented: 'A', 'B', 'AB', 'alternating'
    positional_embedding_scheme = 'hard-alibi' # Implemented: 'alibi', 'hard-alibi', 'abs', 'nope'
    exclude_mlp = False # Whether to exclude the mlp component in the transformer block

    # backbone - pfn configs
    pfn_enc_len = 8

    # icl configs
    train_with_icl = False
    num_icl_examples = 128
    icl_k = 2
    use_query_seq_as_icl = False
    use_random_selection_for_icl = False
    use_transition_matrix_for_icl = False
    weighted_icl_sampling = True
    weighted_decay_rate = 1.0
    truncated_icl_dataset = False

    save_model = True
    verbose = True

    # eval configs
    cold_start_eval = False
    save_metrics = False
    metrics = ['hr', 'mrr'] # Implemented: 'hr', 'mrr'
    eval_k = [3,5,10]

    # synthetic dataset configs
    use_curricular_sdg = False
    # to be filled with lists of variable values
    curricular_sdg_changeable_variables = {'start': None, 'end': None, 'steps': None}

    def __init__(self, **kwargs):

        self.__dict__.update(kwargs)

    def to_dict(self):
        """
        Convert the configuration to a dictionary
        """
        d = {
                key: value for key, value in Config.__dict__.items()
                if not key.startswith('__') and not key.startswith('_') # Exclude dunder and private attributes
                and not callable(value) # Exclude methods
            }
        d.update({key: value for key, value in self.__dict__.items()})
        return d
