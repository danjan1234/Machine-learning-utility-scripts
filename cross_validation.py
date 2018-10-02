"""
Author: Justin Duan
Time: 2018/09/11 10:46AM

This module provides a few tools that help with cross-validation: such k-folds train/validation sets split; 
hyper-parameter scan with cross-validation 
"""

from sklearn.model_selection import KFold
import numpy as np
import types


class CrossValidation(object):
    """
    A class with a few tools (class methods) that help with cross-validation
    """
    @classmethod
    def kfolds(cls, array, n_splits=3,  shuffle=True, random_state=None):
        """
        Split data to k folds, return a generator of train/validation indices.
        :param array: input array. Note the slicing is performed along the first dimension. If array is a pd.DataFrame
               or pd.Series, make sure its index is integer based and starts from 0 to end
        :param n_splits: number of splits
        :param shuffle: boolean. Default=True. Whether to shuffle the data before splitting into batches
        :param random_state: int.RandomState instance. If int, random_state is the seed used by the random number
               generator
        :return: the generator that yields train/validation indices
        """
        if not hasattr(array, '__iter__'):
            raise ValueError("An iterable input data structure is expected.")

        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        return kf.split(array)

    @classmethod
    def kfolds_by(cls, by_array, n_splits=3, shuffle=False, random_state=None):
        """
        Split data to k folds, return a generator of train/validation indices. Note the difference from the basic
        version kfolds is that it takes into consideration the values of the input array, i.e., the split sampling is
        performed with equal probability across all buckets in the input array.
        :param by_array: input array. Note the slicing is performed along the first dimension. If array is a
               pd.DataFrame or pd.Series, make sure its index is integer based and starts from 0 to end
        :param n_splits: number of splits
        :param shuffle: boolean. Default=True. Whether to shuffle the data before splitting into batches
        :param random_state: int.RandomState instance. If int, random_state is the seed used by the random number
               generator
        :return: the generator that yields train/validation indices
        """
        if not hasattr(by_array, '__iter__'):
            raise ValueError("An iterable input data structure is expected.")

        # A nested function to divive the array by its unique values
        def _by_gen(index_array, by_array):
            for x in np.unique(by_array):
                yield index_array[by_array == x]

        kf = KFold(n_splits, shuffle=shuffle, random_state=random_state)

        # Create an array of integer indices of the same length as the input by_array
        index_array = np.arange(0, len(by_array), 1)

        # Create a k-fold splitting generator for each group
        kf_gens = []
        for group in _by_gen(index_array, by_array):
            kf_gens.append(kf.split(group))
        for i in range(n_splits):
            train_indices, test_indices = [], []
            for group, kf_gen in zip(_by_gen(index_array, by_array), kf_gens):
                curr_train_indices, curr_test_indices = next(kf_gen)
                train_indices += list(group[curr_train_indices])
                test_indices += list(group[curr_test_indices])
            yield train_indices, test_indices

    @classmethod
    def cross_validation(cls, estimator, kfolds, X, y):
        """
        Cross validate the input estimator using k-folds train-test sets
        :param estimator: machine learn estimator in comply with Sklearn API requirement
        :param kfolds: a list (or generator) of train/validation indices
        :param X: input X
        :param y: input y
        :return dict. Return mean and std of the train and validation scores
        """
        dic = {}
        train_scores = []
        val_scores = []

        for train_indices, val_indices in kfolds:
            if len(y.shape) == 1:
                (X_train, y_train, X_val, y_val) = (X[train_indices, :], y[train_indices], X[val_indices, :],
                                                    y[val_indices])
            else:
                (X_train, y_train, X_val, y_val) = (X[train_indices, :], y[train_indices, :], X[val_indices, :],
                                                    y[val_indices, :])
            estimator.fit(X_train, y_train)
            train_scores.append(estimator.score(X_train, y_train))
            val_scores.append(estimator.score(X_val, y_val))

        dic['train_score_mean'], dic['train_score_std'] = np.mean(train_scores), np.std(train_scores)
        dic['val_score_mean'], dic['val_score_std'] = np.mean(val_scores), np.std(val_scores)

        return dic

    @classmethod
    def cross_validation_scan(cls, estimator, paras, para_vals, kfolds, X, y):
        """
        Scan the parameter list and perform train and cross-validation for each value.
        :param estimator: machine learn estimator in comply with Sklearn API requirement
        :param paras: list of hyper-parameters
        :param para_vals: list of hyper-parameter values
        :param kfolds: a list (or generator) of train/validation indices
        :param X: input X
        :param y: input y
        :return dict. Return mean and std of the train and validation scores
        """
        if len(paras) != len(para_vals):
            raise  ValueError("Parameter list and parameter value list must have the same length.")
        if isinstance(kfolds, types.GeneratorType):
            kfolds = list(kfolds)   # Preserve the indices
        if not isinstance(kfolds, list):
            raise TypeError("Generator of list types expected for kfolds.")
        if len(kfolds) == 0:
            raise ValueError("The train/cross-validation indices are empty.")

        scores = []
        cls._cross_validation_scan_dfs([], estimator, paras, para_vals, kfolds, X, y, scores)
        return scores

    @classmethod
    def _cross_validation_scan_dfs(cls, indices, estimator, paras, para_vals, kfolds, X, y, scores):
        """
        Helper function for cross_validation. Implemented with depth-first search algorithm.
        """
        if len(indices) == len(paras):
            kwargs = {}
            for para, para_val, i in zip(paras, para_vals, indices):
                kwargs[para] = para_val[i]
            try:
                estimator.set_params(**kwargs)
                score_dic = cls.cross_validation(estimator, kfolds, X, y)
            except AttributeError:
                score_dic = {None}
            scores.append({**kwargs, **score_dic})
            return

        for i in range(len(para_vals[len(indices)])):
            indices.append(i)
            cls._cross_validation_scan_dfs(indices, estimator, paras, para_vals, kfolds, X, y, scores)
            indices.pop()


if __name__ == '__main__':
    print('Unit tests')
    print(CrossValidation.cross_validation_scan(None, ['a', 'b'], [[1, 2], [0.1, 0.2, 0.3]], [1, 2, 3], [1, 1, 1], [1, 1, 1]))
