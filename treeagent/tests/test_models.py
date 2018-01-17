"""
Test for regression models.
"""

import random
import unittest

import numpy as np

from treeagent.models import TreeLeaf, TreeBranch, Ensemble, FastEnsemble

class TestTree(unittest.TestCase):
    """
    Tests for regression trees.
    """
    def test_apply(self):
        """
        Test a tree on a single input at a time.
        """
        tree = _testing_tree()
        for test_in, test_out in _testing_tree_data():
            actual = tree.apply(test_in)
            self.assertTrue(np.allclose(test_out, actual))

    def test_apply_batch(self):
        """
        Test a tree on a batch of inputs at a time.
        """
        tree = _testing_tree()
        data = _testing_tree_data()
        batch_ins, batch_outs = zip(*data)
        actual = tree.apply_batch(batch_ins)
        self.assertTrue(np.allclose(actual, np.array(batch_outs)))

class TestEnsemble(unittest.TestCase):
    """
    Tests for regression ensembles.
    """
    def test_apply(self):
        """
        Test an ensemble on a single input at a time.
        """
        ensemble = _testing_ensemble(dtype=self._input_dtype())
        data = _testing_ensemble_data(ensemble, dtype=self._input_dtype())
        wrapped = self._wrap_ensemble(ensemble)
        for test_in, test_out in data:
            out = wrapped.apply(test_in)
            self.assertTrue(np.allclose(test_out, out))

    def test_apply_batch(self):
        """
        Test an ensemble on batches of inputs.
        """
        ensemble = _testing_ensemble(dtype=self._input_dtype())
        outs = _testing_ensemble_data(ensemble, dtype=self._input_dtype())
        test_ins, test_outs = zip(*outs)
        actual = self._wrap_ensemble(ensemble).apply_batch(test_ins)
        self.assertTrue(np.allclose(actual, np.array(test_outs)))

    # pylint: disable=R0201
    def _wrap_ensemble(self, ensemble):
        """
        Override this to manipulate the ensemble in some
        way and make sure it still does the same thing.
        """
        return ensemble

    # pylint: disable=R0201
    def _input_dtype(self):
        """
        Override this to use a different datatype.
        """
        return 'float64'

class TestFastEnsembleFloat(TestEnsemble):
    """
    Tests for fast ensembles on floating-point data.
    """
    def _wrap_ensemble(self, ensemble):
        return FastEnsemble(ensemble)

class TestFastEnsembleUint8(TestFastEnsembleFloat):
    """
    Tests for fast ensembles on uint8 data.
    """
    def _input_dtype(self):
        return 'uint8'

def test_large_ensemble_apply(benchmark):
    """
    Benchmark stepping a large ensemble.
    """
    ensemble = _testing_ensemble(num_trees=1000)
    inputs, _ = zip(*_testing_ensemble_data(ensemble, num_samples=8))
    benchmark(ensemble.apply_batch, inputs)

def test_fast_ensemble_apply(benchmark):
    """
    Benchmark stepping a large FastEnsemble.
    """
    ensemble = _testing_ensemble(num_trees=1000)
    inputs, _ = zip(*_testing_ensemble_data(ensemble, num_samples=8))
    benchmark(FastEnsemble(ensemble).apply_batch, inputs)

def test_fast_ensemble_create(benchmark):
    """
    Benchmark construction of FastEnsembles.
    """
    ensemble = _testing_ensemble(num_trees=1000)
    benchmark(lambda: FastEnsemble(ensemble))

def _testing_tree():
    """
    Generate tree to test on.
    """
    branch1 = TreeBranch(1, 0.5, TreeLeaf([1, 2.5, 3]), TreeLeaf([2, 3, 1]))
    return TreeBranch(0, 3, TreeLeaf([-3, -5, 3]), branch1)

def _testing_tree_data():
    """
    Generate a sequence of (input, output) pairs for the
    test tree.
    """
    inputs = [[4, 0], [4, 1], [3, 1], [2, 0], [2, 1]]
    outputs = [[1, 2.5, 3], [2, 3, 1], [2, 3, 1], [-3, -5, 3], [-3, -5, 3]]
    return zip(inputs, outputs)

def _testing_ensemble(dtype='float64', num_trees=15):
    """
    Generate an ensemble of random trees.
    """
    weights = []
    trees = []
    for _ in range(num_trees):
        trees.append(_random_tree(dtype))
        weights.append(np.random.normal())
    return Ensemble(np.random.normal(size=(10,)), weights=weights, models=trees)

def _testing_ensemble_data(ensemble, dtype='float64', num_samples=1000):
    """
    Generate a sequence of (input, output) pairs for the
    test ensemble.
    """
    inputs, outputs = [], []
    for _ in range(num_samples):
        an_input = np.random.normal(size=(10,))
        if dtype == 'uint8':
            an_input = np.random.randint(0, high=0x100, size=(10,), dtype=dtype)
        an_output = ensemble.constant_term.copy()
        for weight, model in zip(ensemble.weights, ensemble.models):
            an_output += weight * model.apply(an_input)
        inputs.append(an_input)
        outputs.append(an_output)
    return zip(inputs, outputs)

def _random_tree(dtype, num_outs=10, max_depth=4):
    """
    Create a random tree.
    """
    if max_depth == 0 or random.random() < 0.5:
        return TreeLeaf(np.random.normal(size=(num_outs,)))
    cutoff = np.random.normal()
    if dtype == 'uint8':
        cutoff = random.randrange(0x100)
    return TreeBranch(random.randrange(num_outs), cutoff,
                      _random_tree(dtype, num_outs=num_outs, max_depth=max_depth-1),
                      _random_tree(dtype, num_outs=num_outs, max_depth=max_depth-1))

if __name__ == '__main__':
    unittest.main()
