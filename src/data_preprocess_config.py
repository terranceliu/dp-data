import numpy as np

"""
By default, the domain of each categorical variable is set to be the unique values found in the data.
Alternatively, one can explicitly pass in the domain values in `mappings_cat`

Numerical/continuous attributes are preprocessed based on the dictionary `mappings_num_discretized`
Usage:
    attr: num_bins (int)
        creates `num_bins` bins that are equally spaced across the min and max values for that attribute
    attr: bins (sorted list of ints)
        creates bins of the format
            bins=[a, b, c, d] -> bins = [a, b), [b, c), [c, d)
        if there is repeat (i.e., [a, a]), then we have
            bins=[a, a, b, c, d] -> [a, a], (a, b), [b, c), [c, d)
if an attribute in `attrs_num_discretized` is missing from `mappings_num_discretized`, it will default to
    attr: num_bins=10
"""

class DataPreprocessingConfig():
    def __init__(self, config):
        self.attrs_cat = config['attrs_cat']
        self.attrs_num = config['attrs_num']
        self.attrs_num_discretized = config['attrs_num_discretized']
        self.mappings_cat = config['mappings_cat']
        self.mappings_num = config['mappings_num']
        self.mappings_num_discretized = config['mappings_num_discretized']

    @staticmethod
    def initialize(attrs_cat=None, attrs_num=None, attrs_num_discretized=None, 
                   mappings_cat=None, mappings_num=None, mappings_num_discretized=None):
        config = {}
        config['attrs_cat'] = attrs_cat if attrs_cat is not None else []
        config['attrs_num'] = attrs_num if attrs_num is not None else []
        config['attrs_num_discretized'] = attrs_num_discretized if attrs_num_discretized is not None else []
        config['mappings_cat'] = mappings_cat if mappings_cat is not None else {}
        config['mappings_num'] = mappings_num if mappings_num is not None else {}
        config['mappings_num_discretized'] = mappings_num_discretized if mappings_num_discretized is not None else {}
        
        return DataPreprocessingConfig(config)

def get_config_from_json(attrs_dict):
    attrs_cat, attrs_num, attrs_num_discretized = None, None, None
    mappings_cat, mappings_num, mappings_num_discretized = None, None, None

    attrs_cat = attrs_dict['categorical']
    attrs_num = attrs_dict['numerical']

    preprocessor = DataPreprocessingConfig.initialize(attrs_cat=attrs_cat, 
                                                      attrs_num=attrs_num, 
                                                      attrs_num_discretized=attrs_num_discretized,
                                                      mappings_cat = mappings_cat,
                                                      mappings_num = mappings_num,
                                                      mappings_num_discretized = mappings_num_discretized
                                                      )
    return preprocessor


# elif data_name == 'adult_prediscretized':
#     # categorical
#     attrs_cat = ['workclass', 'education', 'marital-status', 'occupation',
#                     'relationship', 'race', 'sex', 'native-country', 'income>50K']
#     # numerical discretized
#     attrs_num_discretized = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
#     mappings_num_discretized = {'age': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, np.inf],
#                         'capital-gain': [0, 0, 1000, 2000, 3000, 4000, 5000, 7500,
#                                             10000, 15000, 20000, 30000, 50000, np.inf],
#                         'capital-loss': [0, 0, 1000, 2000, 3000, 4000, np.inf],
#                         'hours-per-week': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, np.inf]
#                         }

# elif data_name == 'covtype':
#     assert False, 'not implemented'

# elif data_name == 'epileptic':
#     assert False, 'not implemented'

# elif data_name == 'intrusion':
#     assert False, 'not implemented'

# elif data_name == 'isolet':
#     assert False, 'not implemented'