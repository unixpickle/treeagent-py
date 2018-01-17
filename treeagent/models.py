"""
Tree and forest models.
"""

from abc import ABC, abstractmethod

import numpy as np

# pylint: disable=E0611
from ._models import UnsafeFastEnsemble

# pylint: disable=R0903
class Model(ABC):
    """
    A regression model.
    """
    @abstractmethod
    def apply(self, feature_array):
        """
        Apply the model to a feature array and get an
        output vector (as a numpy.ndarray).

        The output array is owned by the model.
        The caller should not modify it.
        """
        pass

    def apply_batch(self, feature_arrays):
        """
        Apply the model to a batch of feature arrays.

        Returns a 2-D numpy.ndarray with the outer
        dimension as the batch size.
        The caller owns the resulting array.
        """
        return np.array([self.apply(x) for x in feature_arrays], dtype='float32')

    @abstractmethod
    def min_features(self):
        """
        Get the minimum size for inputs to this model.
        This can be used for bounds checking.
        """
        pass

class Ensemble(Model):
    """
    An weighted ensemble of models.
    """
    def __init__(self, constant_term, weights=None, models=None):
        assert isinstance(constant_term, np.ndarray)
        self.constant_term = np.array(constant_term, dtype='float32')
        self.weights = weights or []
        self.models = models or []
        assert len(self.weights) == len(self.models)

    def apply(self, feature_array):
        weighted_sum = self.constant_term.copy()
        for weight, model in zip(self.weights, self.models):
            weighted_sum += model.apply(feature_array) * weight
        return weighted_sum

    def min_features(self):
        if not self.models:
            return 0
        return max([m.min_features() for m in self.models])

    def scale(self, scalar):
        """
        Scale the ensemble by a constant.
        """
        self.constant_term *= scalar
        self.weights = [x * scalar for x in self.weights]

    def extend(self, other_ensemble):
        """
        Add another ensemble to the end of this one.
        """
        self.constant_term += other_ensemble.constant_term
        self.weights.extend(other_ensemble.weights)
        self.models.extend(other_ensemble.models)

    def truncate(self, max_models):
        """
        Truncate the ensemble by removing models from the
        beginning until there are no more than max_models.
        """
        if len(self.models) > max_models:
            diff = len(self.models) - max_models
            self.weights = self.weights[diff:]
            self.models = self.models[diff:]

class TreeBranch(Model):
    """
    A tree node with a feature-based split.
    """
    def __init__(self, split_feature, threshold, less_than, greater_equal):
        """
        Create a branch.

        If the feature indexed by split_feature is less
        than the threshold, forward the decision to the
        less_than branch; otherwise, use greater_equal.
        """
        self.split_feature = split_feature
        self.threshold = float(threshold)
        self.less_than = less_than
        self.greater_equal = greater_equal

    def apply(self, feature_array):
        if feature_array[self.split_feature] < self.threshold:
            return self.less_than.apply(feature_array)
        return self.greater_equal.apply(feature_array)

    def min_features(self):
        return max(self.split_feature+1, self.less_than.min_features(),
                   self.greater_equal.min_features())

class TreeLeaf(Model):
    """
    A tree leaf node with a constant output.
    """
    def __init__(self, output):
        self.output = np.array(output, dtype='float32')

    def apply(self, feature_array):
        return self.output

    def min_features(self):
        return 0

class FastEnsemble(Model):
    """
    A read-only decision tree ensemble that can be
    evaluated quickly.

    Only supports ensembles comprised entirely of TreeLeaf
    and TreeBranch objects.
    """
    def __init__(self, ensemble):
        _validate_ensemble(ensemble)
        self._min_features = ensemble.min_features()
        self._ensemble = UnsafeFastEnsemble(ensemble)

    def apply(self, feature_array):
        if not isinstance(feature_array, np.ndarray):
            feature_array = np.array(feature_array)
        assert len(feature_array.shape) == 1
        if len(feature_array) < self._min_features:
            raise IndexError('feature out of bounds')
        return self._ensemble.apply(feature_array)

    def apply_batch(self, feature_arrays):
        if not isinstance(feature_arrays, np.ndarray):
            feature_arrays = np.array(feature_arrays)
        assert len(feature_arrays.shape) == 2
        if feature_arrays.shape[1] < self._min_features:
            raise IndexError('feature out of bounds')
        return self._ensemble.apply_batch(feature_arrays)

    def min_features(self):
        return self._min_features

def _validate_ensemble(ensemble):
    """
    Make sure that an Ensemble is safe to use with an
    UnsafeFastEnsemble.
    """
    if not isinstance(ensemble, Ensemble):
        raise TypeError('unsupported ensemble type: ' + str(type(Ensemble)))
    out_size = len(ensemble.constant_term)
    if ensemble.constant_term.dtype != 'float32':
        raise TypeError('unsupported output dtype: ' + ensemble.constant_term.dtype)
    if len(ensemble.models) != len(ensemble.weights):
        raise ValueError('mismatching models and weights')
    for model in ensemble.models:
        _validate_ensemble_tree(out_size, model)

def _validate_ensemble_tree(out_size, tree):
    """
    Make sure that a tree from an ensemble is safe to use
    with an UnsafeFastEnsemble.
    """
    if isinstance(tree, TreeBranch):
        for child in [tree.less_than, tree.greater_equal]:
            _validate_ensemble_tree(out_size, child)
    elif isinstance(tree, TreeLeaf):
        if len(tree.output) != out_size:
            raise ValueError('inconsistent output sizes')
        if tree.output.dtype != 'float32':
            raise TypeError('unsupported output dtype: ' + tree.output.dtype)
    else:
        raise TypeError('unsupported model type: ' + str(type(tree)))
