"""
High-level APIs for building trees.
"""

from math import ceil
from multiprocessing import cpu_count
from threading import Thread, Event

import numpy as np

from .models import TreeBranch, TreeLeaf

# pylint: disable=E0611,E0401
from ._build import optimal_feature

# pylint: disable=R0913
def build_tree(inputs,
               outputs,
               min_leaf=1,
               max_depth=3,
               feature_frac=1,
               num_threads=cpu_count()):
    """
    Build a regression tree that maps the inputs to the
    outputs as closely as possible (in the MSE sense).

    Arguments:
      inputs: list-like object of array-like feature maps.
      outputs: list-like object of array-like outputs.
      min_leaf: minimum number of representative samples
        for a leaf.
      max_depth: maximum depth, where one leaf node is
        depth 0.
      feature_frac: the fraction of features to try for
        each split.
      num_threads: maximum number of threads to use for
        parallelizing the computation.

    Returns:
      A Model that is either a leaf or a branch.
    """
    if isinstance(outputs, np.ndarray) and outputs.dtype == 'float32':
        outputs_array = outputs
    else:
        outputs_array = np.array(outputs, dtype='float32')
    assert len(outputs_array.shape) == 2

    if max_depth == 0:
        return _build_leaf(outputs_array)

    if isinstance(inputs, np.ndarray):
        inputs_array = inputs
    else:
        inputs_array = np.array(inputs)
    assert len(inputs_array.shape) == 2

    pool = _SplitterPool(num_threads)
    pool.start()
    try:
        return _build_tree(pool,
                           inputs_array,
                           outputs_array,
                           min_leaf,
                           max_depth,
                           feature_frac)
    finally:
        pool.close()

# pylint: disable=R0913
def _build_tree(pool, inputs, outputs, min_leaf, max_depth, feature_frac):
    """
    Recursive implementation of build_tree().
    """
    if max_depth == 0:
        return _build_leaf(outputs)

    by_feature = np.transpose(inputs)
    use_indices = np.random.choice(len(by_feature),
                                   size=ceil(feature_frac * len(by_feature)),
                                   replace=False).astype('int32')

    best_feature, best_split = pool.optimal_feature(by_feature, outputs,
                                                    use_indices, min_leaf)

    if best_feature is None:
        return _build_leaf(outputs)

    sub_trees = [_build_tree(pool,
                             inputs[best_split[side]],
                             outputs[best_split[side]],
                             min_leaf,
                             max_depth-1,
                             feature_frac)
                 for side in ['less', 'greater']]
    return TreeBranch(best_feature, best_split['threshold'],
                      sub_trees[0], sub_trees[1])

def _build_leaf(outputs_array):
    """
    Build a leaf node.
    """
    return TreeLeaf(np.mean(outputs_array, axis=0))

# pylint: disable=R0902
class _SplitterPool:
    """
    A pool of threads that work together to build trees.
    """
    def __init__(self, num_threads):
        if num_threads == 0:
            self._threads = None
            return
        self._threads = [Thread(target=self._worker, args=(i,))
                         for i in range(num_threads)]
        self._req_events = [Event() for _ in self._threads]
        self._resp_events = [Event() for _ in self._threads]

        self._per_thread_indices = []
        self._per_thread_results = []
        self._current_inputs = None
        self._current_outputs = None
        self._current_min_leaf = None

    def start(self):
        """
        Start running the worker threads.
        """
        if self._threads is None:
            return
        for thread in self._threads:
            thread.start()

    def close(self):
        """
        Close the worker threads.
        """
        if self._threads is None:
            return
        self._per_thread_indices = [None for _ in self._threads]
        for evt in self._req_events:
            evt.set()
        for thread in self._threads:
            thread.join()

    def optimal_feature(self, inputs, outputs, use_indices, min_leaf):
        """
        Find the optimal feature split.

        Behaves like optimal_feature.
        """
        if self._threads is None:
            return optimal_feature(inputs, outputs, use_indices, min_leaf)
        self._current_inputs = inputs
        self._current_outputs = outputs
        self._current_min_leaf = min_leaf
        self._per_thread_indices = _divide_up_work(use_indices, len(self._threads))
        self._per_thread_results = [None] * len(self._per_thread_indices)
        for i in range(len(self._per_thread_indices)):
            self._req_events[i].set()
        best_feature, best_split = None, {}
        for i in range(len(self._per_thread_indices)):
            evt = self._resp_events[i]
            evt.wait()
            evt.clear()
            feature, split = self._per_thread_results[i]
            if split is not None:
                if best_feature is None or split['loss'] < best_split['loss']:
                    best_feature, best_split = feature, split
        return best_feature, best_split

    def _worker(self, index):
        """
        Background routine for worker.
        """
        evt = self._req_events[index]
        resp_evt = self._resp_events[index]
        while True:
            evt.wait()
            evt.clear()
            indices = self._per_thread_indices[index]
            if indices is None:
                return
            result = optimal_feature(self._current_inputs,
                                     self._current_outputs,
                                     indices,
                                     self._current_min_leaf)
            self._per_thread_results[index] = result
            resp_evt.set()

def _divide_up_work(work, num_workers):
    """
    Divide the work (a list of values) into at most
    num_workers sub-lists which are as balanced as
    possible.
    """
    common_size = len(work) // num_workers
    extra_size = len(work) % num_workers
    res = []
    for i in range(num_workers):
        if i < extra_size:
            res.append(work[i*(common_size+1):(i+1)*(common_size+1)])
        elif common_size > 0:
            extra_off = extra_size + common_size*extra_size
            cur_off = (i-extra_size)*common_size + extra_off
            res.append(work[cur_off:cur_off+common_size])
    return res
