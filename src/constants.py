import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


SDG_STAGE_1_PARAMS = \
    {
        'transition_matrix_prior': ['random_graph'],
        'popularity_bias': [0.0, 0.1, 0.2, 0.3, 0.5],
        'window_size': [1, 2, 3],
        'decay_coeff': [0.5, 0.8],
        'repeated_items': [True],
    }

SDG_STAGE_2_PARAMS = \
    {
        'transition_matrix_prior': ['random_graph', 'latent_factor'],
        'num_latent_concepts': [20,40,60],
        'mean_num_concepts_per_item': [2, 3, 5],
        'mean_num_concepts_per_user': [2, 3, 5],
        'popularity_bias': [0.0, 0.1, 0.2],
        'window_size': [1,2,3],
        'decay_coeff': [0.5, 0.8],
        'logit_scale': [9.0, 12.0],
        'concept_matrix_diagonal_deviation_rate': [0.0, 0.1, 0.3],
        'repeated_items': [True],
        'mean_num_related_items': [1,3,5,7],
        'repeated_item_probability': [0.1, 0.2, 0.3],
    }
