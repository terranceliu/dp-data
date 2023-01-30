import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def get_bins( values, num_bins):
    bin_ranges = np.linspace(values.min(), values.max(), num_bins + 1)
    bin_ranges[0] = -np.inf
    bin_ranges[-1] = np.inf
    return bin_ranges.tolist()

"""
fill_missing: If True, missing categorical values are mapped to _OTHER
default_num_bins: If mappings_num_discretized[attr] is not specified, attr will be mapped to `default_num_bins`.
"""
class DataPreprocessor():
    def __init__(self, config, fill_missing: bool=False, default_num_bins: int=10):
        self.config = config
        self.fill_missing = fill_missing
        self.default_num_bins = default_num_bins

        self.attrs_cat = config.attrs_cat
        self.attrs_num = config.attrs_num
        self.attrs_num_discretized = config.attrs_num_discretized
        self.attrs = self.attrs_cat + self.attrs_num + self.attrs_num_discretized
        assert len(self.attrs) > 0, 'must input at least one attribute'

        self.mappings_cat = config.mappings_cat
        self.mappings_num = config.mappings_num
        self.mappings_num_discretized = config.mappings_num_discretized

        self.encoders = {}

        # Require at least 2 bins
        assert self.default_num_bins >= 2, 'default_num_bins must be >= 2'
        for attr, bins in self.mappings_num_discretized.items():
            assert len(bins) >= 3, f'{attr} must have >=3 bins'

    def _get_df_domain(self, df):
        for attr in self.attrs_cat:
            if attr in self.mappings_cat.keys():
                if self.fill_missing:
                    self.mappings_cat[attr].append('_OTHER')
            else:
                self.mappings_cat[attr] = df[attr].unique().tolist()

        num_rows = max([len(x) for x in self.mappings_cat.values()])
        df_domain = df.loc[:num_rows].copy()
        if len(df_domain) < num_rows:
            factor = np.ceil(num_rows / len(df_domain)).astype(int)
            df_domain = pd.concat([df_domain] * factor).reset_index(drop=True)
        for attr, categories in self.mappings_cat.items():
            df_domain.loc[:len(categories) - 1, attr] = categories
            df_domain.loc[len(categories) + 1:, attr] = categories[0]

        return df_domain

    def get_domain(self):
        domain = {}
        for attr in self.attrs_cat:
            domain[attr] = len(self.encoders[attr].classes_)
            assert domain[attr] >= 2, f'{attr} takes on only value'
        for attr in self.attrs_num:
            domain[attr] = 1
        for attr in self.attrs_num_discretized:
            domain[attr] = len(self.mappings_num_discretized[attr]) - 1
            assert domain[attr] >= 2, 'f{attr} takes on only value'
        return domain

    ##### categorical #####
    def fit_cat(self, df):
        df = self._get_df_domain(df)
        for attr in self.attrs_cat:
            enc = LabelEncoder()
            enc.fit(df[attr].values)
            self.encoders[attr] = enc

    def transform_cat(self, df):
        for attr, categories in self.mappings_cat.items():
            mask = ~df[attr].isin(categories)
            if mask.sum() > 0:
                if self.fill_missing and '_OTHER' in categories:
                    df.loc[mask, attr] = '_OTHER'
                else:
                    assert False, 'invalid value found in data (attr: {})'.format(attr)

        for attr in self.attrs_cat:
            enc = self.encoders[attr]
            encoded = enc.transform(df[attr].values)
            df.loc[:, attr] = encoded

    def inverse_transform_cat(self, df):
        for attr in self.attrs_cat:
            enc = self.encoders[attr]
            df.loc[:, attr] = enc.inverse_transform(df[attr].values)

    ##### numerical #####
    def fit_num(self, df):
        for attr in self.attrs_num:
            if attr not in self.mappings_num.keys():
                self.mappings_num[attr] = df[attr].min(), df[attr].max()
            else:
                max_val, min_val = self.mappings_num[attr]
                assert df[attr].min() < min_val, f'{attr}: invalid config min value'
                assert df[attr].max() > max_val, f'{attr}: invalid config max value'

    def transform_num(self, df):
        for attr, (min_val, max_val) in self.mappings_num.items():
            df.loc[:, attr] = (df[attr].values - min_val) / (max_val - min_val)
    
    def inverse_transform_num(self, df):
        for attr, (min_val, max_val) in self.mappings_num.items():
            df.loc[:, attr] = df[attr].values * (max_val - min_val) + min_val

    ##### numerical (discretized) #####
    def fit_num_discretized(self, df):
        for attr in self.attrs_num_discretized:
            bin_ranges = get_bins(df[attr].values, self.default_num_bins)
            if attr in self.mappings_num_discretized.keys():
                if isinstance(self.mappings_num_discretized[attr], list):
                    bin_ranges = self.mappings_num_discretized[attr]
                    assert sorted(bin_ranges) == bin_ranges, '`bin_ranges` must be sorted.'
                elif isinstance(self.mappings_num_discretized[attr], int):
                    num_bins = self.mappings_num_discretized[attr]
                    bin_ranges = self._get_bins(df[attr].values, num_bins)
                else:
                    assert False, 'invalid config entry for {}'.format(attr)
            assert df[attr].min() >= bin_ranges[0], 'min value in bins is larger than the min value in the data'
            assert df[attr].max() >= bin_ranges[1], 'max value in bins is smaller than the max value in the data'

            self.mappings_num_discretized[attr] = bin_ranges

    def transform_num_discretized(self, df):
        for attr, bin_ranges in self.mappings_num_discretized.items():
            output = df[attr].copy()
            for i in range(len(bin_ranges) - 1):
                lower = float(bin_ranges[i])
                upper = float(bin_ranges[i + 1])
                if lower == upper:
                    mask = df[attr] == lower
                elif i > 0 and bin_ranges[i - 1] == lower:
                    mask = (df[attr] >= lower) & (df[attr] < upper)
                else:
                    mask = (df[attr] >= lower) & (df[attr] < upper)
                output[mask] = i
            df.loc[:, attr] = output.astype(int)

    def inverse_transform_num_discretized(self, df):
        for attr, bin_ranges in self.mappings_num_discretized.items():
            output = df[attr].copy()
            for i in range(len(bin_ranges) - 1):
                lower = bin_ranges[i]
                upper = bin_ranges[i + 1]
                if lower == upper:
                    val = str(lower)
                elif i > 0 and bin_ranges[i - 1] == lower:
                    val = '({}, {})'.format(lower, upper)
                else:
                    val = '[{}, {})'.format(lower, upper)
                output[df[attr] == i] = val
            df.loc[:, attr] = output

    ##### transform functions ######
    def fit(self, df):
        df.reset_index(drop=True, inplace=True)
        self.fit_cat(df)
        self.fit_num(df)
        self.fit_num_discretized(df)

    def transform(self, df):
        df = df.loc[:, self.attrs].copy()
        self.transform_cat(df)
        self.transform_num(df)
        self.transform_num_discretized(df)
        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, df):
        df = df.copy()
        self.inverse_transform_cat(df)
        self.inverse_transform_num(df)
        self.inverse_transform_num_discretized(df)
        return df