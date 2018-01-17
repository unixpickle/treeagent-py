"""
Optimized models.
"""

import cython
from libc.stdlib cimport calloc, free

import numpy as np
cimport numpy as np

np.import_array()

cdef class UnsafeFastEnsemble:
    """
    A fast C-based ensemble with no safety checks.

    Before constructing an instance, you must verify that
    the ensemble has consistent output vector lengths,
    valid models (TreeLeaf or TreeBranch), etc.
    """
    cdef int num_trees
    cdef int out_size
    cdef _FastTree** trees
    cdef float* weights
    cdef float* constant

    def __cinit__(self, ensemble):
        self.num_trees = len(ensemble.models)
        self.out_size = len(ensemble.constant_term)
        self.trees = <_FastTree**>calloc(self.num_trees, sizeof(_FastTree*))
        self.weights = <float*>calloc(self.num_trees, sizeof(float))
        self.constant = _copy_vector(ensemble.constant_term)
        for i, (weight, tree) in enumerate(zip(ensemble.weights, ensemble.models)):
            self.weights[i] = weight
            self.trees[i] = _make_fast_tree(tree)

    def __dealloc__(self):
        cdef int i
        for i in range(self.num_trees):
            _free_fast_tree(self.trees[i])
        free(self.trees)
        free(self.weights)
        free(self.constant)

    def apply(UnsafeFastEnsemble self, np.ndarray input):
        """
        Applying the ensemble to the feature map.

        This does not perform bounds checking.

        Returns:
          A numpy.ndarray of float32 values.
        """
        result = np.zeros(self.out_size, dtype='float32')
        _weighted_add(result, self.constant, self.out_size, 1)
        if input.dtype == 'uint8':
            self._apply_uint8(result, input)
        elif input.dtype == 'float32':
            self._apply_float(result, input)
        else:
            self._apply_float(result, input.astype('float32'))
        return result

    def apply_batch(UnsafeFastEnsemble self, np.ndarray inputs):
        """
        Apply the ensemble to the feature maps.

        This does not perform bounds checking.

        Returns:
          A numpy.ndarray of float32 values.
        """
        cdef int num_inputs = len(inputs)
        cdef int i
        results = np.zeros((len(inputs), self.out_size), dtype='float32')
        for i in range(num_inputs):
            _weighted_add(results[i], self.constant, self.out_size, 1)
        if inputs.dtype == 'uint8':
            self._apply_uint8s(results, inputs, num_inputs)
        elif inputs.dtype == 'float32':
            self._apply_floats(results, inputs, num_inputs)
        else:
            self._apply_floats(results, inputs.astype('float32'), num_inputs)
        return results

    cdef void _apply_uint8(UnsafeFastEnsemble self, float[:] result,
                           unsigned char[:] features) nogil:
        for i in range(self.num_trees):
            tree_out = _apply_fast_tree_uint8(self.trees[i], features)
            _weighted_add(result, tree_out, self.out_size, self.weights[i])

    cdef void _apply_float(UnsafeFastEnsemble self, float[:] result,
                           float[:] features) nogil:
        for i in range(self.num_trees):
            tree_out = _apply_fast_tree(self.trees[i], features)
            _weighted_add(result, tree_out, self.out_size, self.weights[i])

    cdef void _apply_uint8s(UnsafeFastEnsemble self, float[:, :] results,
                            unsigned char[:, :] features, int num_ins) nogil:
        cdef int i
        for i in range(num_ins):
            self._apply_uint8(results[i], features[i])

    cdef void _apply_floats(UnsafeFastEnsemble self, float[:, :] results,
                            float[:, :] features, int num_ins) nogil:
        cdef int i
        for i in range(num_ins):
            self._apply_float(results[i], features[i])

cdef struct _FastTree:
    int branch_feature
    float branch_threshold
    _FastTree* left_tree
    _FastTree* right_tree
    float* leaf_output

cdef _FastTree* _make_fast_tree(tree):
    """
    Make a fast tree.
    """
    res = <_FastTree *>calloc(1, sizeof(_FastTree))
    if hasattr(tree, 'output'):
        res.leaf_output = _copy_vector(tree.output)
    else:
        res.branch_feature = tree.split_feature
        res.branch_threshold = tree.threshold
        res.left_tree = _make_fast_tree(tree.less_than)
        res.right_tree = _make_fast_tree(tree.greater_equal)
    return res

cdef void _free_fast_tree(_FastTree* tree):
    """
    Free a fast tree.
    """
    if tree.left_tree:
        _free_fast_tree(tree.left_tree)
    if tree.right_tree:
        _free_fast_tree(tree.right_tree)
    if tree.leaf_output:
        free(tree.leaf_output)
    free(tree)

@cython.boundscheck(False)
cdef float* _apply_fast_tree(_FastTree* tree, float[:] features) nogil:
    """
    Apply a fast tree to floating-point features.
    """
    if tree.leaf_output:
        return tree.leaf_output
    else:
        if features[tree.branch_feature] < tree.branch_threshold:
            return _apply_fast_tree(tree.left_tree, features)
        else:
            return _apply_fast_tree(tree.right_tree, features)

@cython.boundscheck(False)
cdef float* _apply_fast_tree_uint8(_FastTree* tree, unsigned char[:] features) nogil:
    """
    Apply a fast tree to uint8 features.
    """
    if tree.leaf_output:
        return tree.leaf_output
    else:
        if <float>features[tree.branch_feature] < tree.branch_threshold:
            return _apply_fast_tree_uint8(tree.left_tree, features)
        else:
            return _apply_fast_tree_uint8(tree.right_tree, features)

cdef float* _copy_vector(numpy_vec):
    """
    Create a float array from a numpy array.
    """
    assert numpy_vec.dtype == np.float32
    cdef int i
    cdef float* vec = <float *>calloc(len(numpy_vec), sizeof(float))
    for i, value in enumerate(numpy_vec):
        vec[i] = numpy_vec[i]
    return vec

@cython.boundscheck(False)
cdef void _weighted_add(float[:] dst, float* src, int len, float weight) nogil:
    """
    Scale src by weight and add it to dst.
    """
    cdef int i
    for i in range(len):
        dst[i] += src[i] * weight
