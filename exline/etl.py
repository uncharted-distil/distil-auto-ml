#!/usr/bin/env python

"""
    etl.py
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import impute, preprocessing
from sklearn_pandas import DataFrameMapper, CategoricalImputer

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict

# --
# Beeline

MISSING_VALUE_INDICATOR = '__miss_salt%d' % np.random.choice(int(1e6))
SINGLETON_INDICATOR     = '__sing_salt%d' % np.random.choice(int(1e6))

class BinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.lookup = None
    
    def fit(self, X):
        levels = list(set(X.squeeze())) + [MISSING_VALUE_INDICATOR] # !! How to sort? Randomly? Alphabetically?
        levels = np.random.permutation(levels)
        
        vals      = range(len(levels))
        max_width = len(np.binary_repr(max(vals)))
        vals      = [np.binary_repr(v, width=max_width) for v in vals]
        vals      = [np.array(list(v)).astype(int) for v in vals]
        
        self.lookup = dict(zip(levels, vals))
        return self
    
    def transform(self, X, y=None):
        assert self.lookup is not None
        return np.vstack([self.lookup.get(xx, self.lookup[MISSING_VALUE_INDICATOR]) for xx in X.squeeze()])
    
    def fit_transform(self, X, y=None, **kwargs):
        _ = self.fit(X)
        return self.transform(X)


class SVMTextEncoder(BaseEstimator, TransformerMixin):
    # !! add tuning
    def __init__(self):
        super().__init__()
        
        self._vect  = TfidfVectorizer(ngram_range=[1, 2], max_features=30000)
        self._model = LinearSVC(class_weight='balanced')
    
    def fit(self, X, y):
        raise NotImplemented
    
    def transform(self, X):
        X = pd.Series(X.squeeze()).fillna(MISSING_VALUE_INDICATOR).values
        
        Xv  = self._vect.transform(X)
        out = self._model.decision_function(Xv)
        
        if len(out.shape) == 1:
            out = out.reshape(-1, 1)
        
        return out
    
    def fit_transform(self, X, y=None, **kwargs):
        assert y is not None, 'SVMTextEncoder.fit_transform requires y'
        
        X = pd.Series(X.squeeze()).fillna(MISSING_VALUE_INDICATOR).values
        
        Xv  = self._vect.fit_transform(X)
        out = cross_val_predict(self._model, Xv, y, method='decision_function', n_jobs=3, cv=3)
        self._model = self._model.fit(Xv, y)
        
        if len(out.shape) == 1:
            out = out.reshape(-1, 1)
        
        return out


def detect_text(X, thresh=8):
    """ returns true if median entry has more than `thresh` tokens"""
    X = X[X.notnull()]
    n_toks = X.apply(lambda xx: len(xx.split(' '))).values
    return np.median(n_toks) >= thresh


def replace_singletons(X_train, X_test, keep_text=True):
    """ set values that only occur once to a special token """
    
    cols = list(X_train.columns)
    for c in cols:
        if X_train[c].dtype == np.object_:
            if not keep_text or not detect_text(X_train[c]):
                vcs        = pd.value_counts(list(X_train[c]) + list(X_test[c]))
                singletons = set(vcs[vcs == 1].index)
                if singletons:
                    X_train[c][X_train[c].isin(singletons)] = SINGLETON_INDICATOR
                    X_test[c][X_test[c].isin(singletons)]   = SINGLETON_INDICATOR
    
    return X_train, X_test


def make_mapper(X, mode='tree', cat_mode='one_hot', max_one_hot=16, max_binary=10000):
    assert mode in ['rgf', 'tree', 'etree', 'linear']
    assert cat_mode in ['one_hot', 'binary']
    
    maps = []
    for c in X.columns:
        if X[c].dtype == np.object_:
            uvals = list(set(X[c]))
            if len(uvals) == 1:
                print('%s has 1 level -> skipping' % c, file=sys.stderr)
                continue
            
            categories = [uvals + [MISSING_VALUE_INDICATOR]]
            
            if detect_text(X[c]):
                print('----- %s has text detected -> use SVMTextEncoder' % c, file=sys.stderr)
                cat_encoder = SVMTextEncoder()
            elif (cat_mode == 'one_hot') and (len(uvals) < max_one_hot):
                cat_encoder = preprocessing.OneHotEncoder(sparse=False, categories=categories, handle_unknown='ignore')
            # elif len(uvals) >= max_binary:
            #     # ... but if it has too many levels, exclude it
            #     print('----- %s has %d levels -> is it continuous? skipping' % (c, len(uvals)), file=sys.stderr)
            #     continue
            else:
                # Binary encoding triggered automatically if variable has enough levels
                print('%s has %d levels -> use BinaryEncoder' % (c, len(uvals)), file=sys.stderr)
                cat_encoder = BinaryEncoder()
            
            maps.append(([c], [
                CategoricalImputer(strategy='constant', fill_value=MISSING_VALUE_INDICATOR),
                cat_encoder,
            ]))
        
        elif X[c].dtype in [float, int]:
            if mode in ['rgf', 'tree', 'etree']:
                maps.append(([c], [
                    impute.SimpleImputer()
                ]))
            elif mode == 'linear':
                maps.append(([c], [
                    impute.SimpleImputer(),
                    preprocessing.StandardScaler(),
                ]))
            
            if X[c].isnull().any():
                maps.append(([c], [
                    impute.MissingIndicator()
                ]))
            
        else:
            raise NotImplemented
    
    return DataFrameMapper(maps)


def apply_mapper(X_train, X_test, mapper, y_train=None):
    Xf_train = mapper.fit_transform(X_train, y_train)
    Xf_test  = mapper.transform(X_test)
    keep = (
        ((Xf_train != 0).sum(axis=0) != 0) & 
        ((Xf_test != 0).sum(axis=0) != 0)
    )
    Xf_train, Xf_test = Xf_train[:,keep], Xf_test[:,keep]
    return Xf_train, Xf_test


def load_ragged_timeseries(X, base_path):
    c = 'timeseries_file' if 'timeseries_file' in X.columns else X.columns[0]
    paths = X[c].apply(lambda x: os.path.join(base_path, 'timeseries', x))
    return [pd.read_csv(p).values[:,1] for p in paths]


def load_timeseries(X, base_path):
    ts = load_ragged_timeseries(X, base_path)
    assert len(set([len(tt) for tt in ts])) == 1 # assume all same length
    return np.vstack(ts)


def load_ragged_sets(X, base_path, colname=None):
    colname = X.columns[0] if colname is None else colname
    paths = X[colname].apply(lambda x: os.path.join(base_path, x))
    return [pd.read_csv(p).values for p in paths]


def load_sets(X, base_path, colname=None):
    ts = load_ragged_sets(X, base_path, colname=colname)
    assert len(set([t.shape for t in ts])) == 1
    return np.stack(ts)


# ---------------------
# Slacker

from collections import defaultdict, OrderedDict

import numpy as np
from scipy import signal
from scipy.sparse import csr_matrix, hstack

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

class DenseMixedStrategyImputer(BaseEstimator, TransformerMixin):

    def __init__(self, missing_values='NaN', strategies=None, add_missing_indicator=True, verbose=False):
        self.missing_values = missing_values
        if strategies is None:
            raise ValueError('Must provide strategy.')
        allowed_strategies = ['mean', 'median', 'most_frequent']
        if any(s not in allowed_strategies for s in strategies):
            raise ValueError('Invalid strategy in list.')
        self.strategies = strategies
        self.add_missing_indicator = add_missing_indicator
        self.verbose = verbose

    def fit(self, X, y=None):
        n_samples, n_features = X.shape
        if len(self.strategies) != n_features:
            raise ValueError('Number of strategies must equal number of features.')
        self.impute_strategies = list(set(self.strategies))
        self.impute_indices = [np.array([i for i, x in enumerate(self.strategies) if x == s]) for s in self.impute_strategies]
        self.impute_valid_indices = []
        self.imputers = [Imputer(missing_values=self.missing_values, strategy=s, verbose=self.verbose) for s in
                         self.impute_strategies]
        for indices, imputer in zip(self.impute_indices, self.imputers):
            imputer.fit(X[:, indices])
            valid_mask = np.logical_not(np.isnan(imputer.statistics_))
            self.impute_valid_indices.append(indices[valid_mask])
        return self

    def transform(self, X):
        n_samples, n_features = X.shape
        if len(self.strategies) != n_features:
            raise ValueError('Number of strategies must equal number of features.')
        check_is_fitted(self, 'imputers')

        if self.add_missing_indicator:
            output_scale = 2
        else:
            output_scale = 1

        X_out = np.zeros((n_samples, output_scale*n_features))
        for input_indices, output_indices, imputer in zip(self.impute_indices, self.impute_valid_indices, self.imputers):
            X_out[:, output_scale*output_indices] = imputer.transform(X[:, input_indices])

        if self.add_missing_indicator:
            X_out[:, np.arange(1, 2*n_features, 2)] = np.isnan(X).astype('float', copy=False)

        return X_out


class DataFrameCategoricalEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.code_maps = {}
        for k in X.columns:
            self.code_maps[k] = defaultdict(lambda: np.nan)
            self.code_maps[k].update({v: k for k, v in enumerate(X[k].astype('category').cat.categories)})
        return self
        
    def transform(self, X):
        if set(X.columns) != set(self.code_maps):
            raise ValueError('Columns do not match fit model.')
        
        return X.apply(lambda x: x.apply(lambda y: self.code_maps[x.name][y])).as_matrix()


class AnnotatedTabularExtractor(BaseEstimator):

    param_distributions = {
        'normalize_text'        : [True, False],
        'categorize'            : [True, False],
        'numeric_strategy'      : ['mean', 'median'],
        'add_missing_indicator' : [True, False]
    }
    
    def __init__(self, normalize_text=False, categorize=False, numeric_strategy='mean', add_missing_indicator=True):
        self.normalize_text        = normalize_text
        self.categorize            = categorize
        self.numeric_strategy      = numeric_strategy
        self.add_missing_indicator = add_missing_indicator
        
    def fit(self, df, variables):
        self.fit_transform(df, variables)
        return self
        
    def fit_transform(self, df, variables):
        df = self.copy_normalize_text(df)
        
        self.column_types = OrderedDict()
        for column in df:
            var = variables[column]
            type = var.type
            series = df[column]
            
            if var.type == 'unknown':
                try:
                    pd.to_numeric(series, errors='raise')
                    type = 'numeric'
                except:
                    if series.str.len().median() >= 20:
                        type = 'text'
                        
            if (self.categorize or var.type == 'unknown') and len(set(series)) < 10 * np.log10(len(series)):
                self.column_types[column] = 'categorical'
                
            elif type in {'categorical', 'boolean'}:
                self.column_types[column] = 'categorical'
                
            elif type == 'text':
                self.column_types[column] = 'text'
                
            else:
                self.column_types[column] = 'numeric'
                
        self.numeric_columns = [column for column, type in self.column_types.items() if type == 'numeric']
        self.categorical_columns = [column for column, type in self.column_types.items() if type == 'categorical']
        self.text_columns = [column for column, type in self.column_types.items() if type == 'text']
        
        output_arrays = []
        
        if len(self.numeric_columns) > 0:
            X = df[self.numeric_columns].apply(lambda x: pd.to_numeric(x, errors='coerce')).as_matrix()
            self.numeric_imputer = DenseMixedStrategyImputer(
                strategies=[self.numeric_strategy]*len(self.numeric_columns),
                add_missing_indicator=self.add_missing_indicator
            )
            X = self.numeric_imputer.fit_transform(X)
            self.numeric_scaler = StandardScaler()
            output_arrays.append(self.numeric_scaler.fit_transform(X))
            
        if len(self.categorical_columns) > 0:
            self.categorical_encoder = DataFrameCategoricalEncoder()
            X = self.categorical_encoder.fit_transform(df[self.categorical_columns])
            self.categorical_imputer = DenseMixedStrategyImputer(
                strategies=['most_frequent']*len(self.categorical_columns),
                add_missing_indicator=self.add_missing_indicator
            )
            X = self.categorical_imputer.fit_transform(X)
            self.one_hot_encoder = OneHotEncoder(
                categorical_features=np.arange(len(self.categorical_columns)) * (2 if self.add_missing_indicator else 1)
            )
            output_arrays.append(self.one_hot_encoder.fit_transform(X))
            
        return hstack([csr_matrix(X) for X in output_arrays], format='csr')
        
    def transform(self, df):
        
        check_is_fitted(self, 'column_types')
        if list(df) != list(self.column_types):
            raise ValueError('Data to be transformed does not match fitting data.')
            
        df = self.copy_normalize_text(df)
        
        output_arrays = []
        
        if len(self.numeric_columns) > 0:
            X = df[self.numeric_columns].apply(lambda x: pd.to_numeric(x, errors='coerce')).as_matrix()
            output_arrays.append(self.numeric_scaler.transform(self.numeric_imputer.transform(X)))
            
        if len(self.categorical_columns) > 0:
            X = self.categorical_encoder.transform(df[self.categorical_columns])
            output_arrays.append(self.one_hot_encoder.transform(self.categorical_imputer.transform(X)))
            
        return hstack([csr_matrix(X) for X in output_arrays], format='csr')

    def copy_normalize_text(self, df):
        df = df.copy()
        if self.normalize_text:
            for column in df:
                df[column] = df[column].str.lower().str.strip()
        return df

    @classmethod
    def supports_variable(cls, d3m_var):
        return None
