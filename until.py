import pandas as pd
import numpy as np

import dask.dataframe as dd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from sklearn.metrics import f1_score, classification_report

from sklearn.feature_selection import SelectFromModel

def undersample_df_by_target(df, target_name):
    num_0 = len(df[df[target_name] == 0])
    num_1 = len(df[df[target_name] == 1])
    undersampled_data = pd.concat([df[df[target_name] == 0].sample(num_1), df[df[target_name] == 1]])

    return undersampled_data


def run_grid_search(estimator, X, y, params_grid, scoring='f1'):
    gsc = GridSearchCV(estimator, params_grid, scoring=scoring, cv=3, n_jobs=-1)

    gsc.fit(X, y)
    print("Best %s score: %.2f" % (scoring, gsc.best_score_))
    print()
    print("Best parameters set found on development set:")
    print()
    print(gsc.best_params_)
    print()
    print("Grid scores on development set:")
    print()

    for i, params in enumerate(gsc.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (gsc.cv_results_['mean_test_score'][i], gsc.cv_results_['std_test_score'][i] * 2, params))

    print()

    return gsc

def treshold_search(y_true, y_pred):
    top = [0.5, f1_score(y_true, y_pred[: , 1] > 0.5, average='macro')]
    for treshold in np.linspace(0, 1, 20):
        fscore = f1_score(y_true, y_pred[: , 1] > treshold, average='macro')
        if fscore > top[1]:
            top[0] = treshold
            top[1] = fscore
    print(f'Лучшая отсечка : {top[0]}, Метрика F1_macro: {top[1]}')
    print("=" * 80)
    print(classification_report(y_true, y_pred[:, 1] > top[0]))


def preprocess_data_train(prep_data_df, FEATURES_DATA):
    prep_data_df['buy_time'] = pd.to_datetime(prep_data_df['buy_time'], unit='s')
    #prep_data_df = prep_data_df.drop('Unnamed: 0', axis=1)
    prep_data_df['monthday'] = prep_data_df['buy_time'].dt.day
    prep_data_df = prep_data_df.sort_values('buy_time')
    prep_data_df['not_first_offer'] = prep_data_df.duplicated('id').astype(int)

    features_data_df = dd.read_csv(FEATURES_DATA, sep='\t')
    features_data_df = features_data_df.drop('Unnamed: 0', axis=1)
    train_list_index = list(prep_data_df['id'].unique())
    features_data_df = features_data_df.loc[features_data_df['id'].isin(train_list_index)].compute()
    features_data_df['buy_time'] = pd.to_datetime(features_data_df['buy_time'], unit='s')
    features_data_df = features_data_df.sort_values(by="buy_time")

    result_data = pd.merge_asof(prep_data_df, features_data_df, on='buy_time', by='id', direction='nearest')

    result_data.drop(['id', 'buy_time'], axis=1, inplace=True)
    result_data.drop_duplicates(inplace=True)
    
    result_data = result_data.set_index(['Unnamed: 0'])
    result_data.index.name = None
    result_data.sort_index(inplace=True)
    
    return result_data, train_list_index


def preprocess_data_test(prep_data_df, FEATURES_DATA, train_list_index):
    prep_data_df['buy_time'] = pd.to_datetime(prep_data_df['buy_time'], unit='s')
    #prep_data_df = prep_data_df.drop('Unnamed: 0', axis=1)
    prep_data_df['monthday'] = prep_data_df['buy_time'].dt.day
    prep_data_df = prep_data_df.sort_values('buy_time')
    prep_data_df['not_first_offer'] = (prep_data_df['id'].isin(train_list_index)).astype(int)

    features_data_df = dd.read_csv(FEATURES_DATA, sep='\t')
    features_data_df = features_data_df.drop('Unnamed: 0', axis=1)
    test_list_index = list(prep_data_df['id'].unique())
    features_data_df = features_data_df.loc[features_data_df['id'].isin(test_list_index)].compute()
    features_data_df['buy_time'] = pd.to_datetime(features_data_df['buy_time'], unit='s')
    features_data_df = features_data_df.sort_values(by="buy_time")

    result_data = pd.merge_asof(prep_data_df, features_data_df, on='buy_time', by='id', direction='nearest')

    result_data.drop(['id', 'buy_time'], axis=1, inplace=True)
    
    result_data = result_data.set_index(['Unnamed: 0'])
    result_data.index.name = None
    result_data.sort_index(inplace=True)
    
    return result_data


def select_type_cols(merged_data):
    X_nunique = merged_data.apply(lambda x: x.nunique(dropna=False))
    f_all = set(X_nunique.index.tolist())
    f_const = set(X_nunique[X_nunique == 1].index.tolist())
    f_categorical = set(X_nunique[X_nunique <= 30].index.tolist())
    f_numeric = (merged_data.fillna(0).astype(int).sum() - merged_data.fillna(0).sum()).abs()
    f_numeric = set(f_numeric[f_numeric > 0].index.tolist())
    f_binary = set(merged_data.loc[:, f_all].columns[(
            (merged_data.loc[:, f_all].max() == 1) & \
            (merged_data.loc[:, f_all].min() == 0) & \
            (merged_data.loc[:, f_all].isnull().sum() == 0))])
    f_categorical = f_categorical - f_const - f_binary
    f_numeric = f_numeric - f_categorical - f_const

    assert (X_nunique.shape[0] == len(f_const) + len(f_binary) + len(f_numeric) + len(f_categorical))

    f_all = list(f_binary | f_categorical | f_numeric)
    f_binary, f_categorical, f_numeric = list(f_binary), list(f_categorical), list(f_numeric)

    return f_all, f_binary, f_categorical, f_numeric


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("DataFrame не содердит следующие колонки: %s" % cols_error)