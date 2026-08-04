class BaseConfig:

    def __init__(self, **kwargs):
        """
        Initialize the configuration with given keyword arguments.
        """
        self.__dict__.update(kwargs)

    def to_dict(self):
        """
        Convert the configuration to a dictionary
        """
        d = {
                key: value for key, value in self.__dict__.items()
                if not key.startswith('__') and not key.startswith('_')
                and not callable(value)
            }
        return d

class SDGConfig(BaseConfig):

    random_seed = 42

    # general dataset params
    variable_seq_len = True
    max_seq_len = 30
    min_seq_len = 5
    repeated_items = False
    transition_matrix_prior = 'random_graph'  # Options: 'random_graph', 'latent_factor'

    # random graph params
    mean_num_related_items = 3
    repeated_item_probability = 0.2

    # latent factor params
    num_latent_concepts = 20
    concept_matrix_occupancy_rate = 0.2
    concept_matrix_diagonal_deviation_rate = 0.5
    mean_num_concepts_per_item = 5
    continuous_concept_weights = False

    # model params
    popularity_bias = 0.2
    noise_coefficient = 0.01
    predefined_popularity = True
    predefined_popularity_coeffient = 4.

    # walk params
    window_size = 3
    decay_coeff = 0.5
    logit_scale = 8.0

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
