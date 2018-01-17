"""
Optimized tree-building routines.
"""

import cython
from libc.stdlib cimport malloc, calloc, free

import numpy as np
cimport numpy as np

np.import_array()

def optimal_feature(np.ndarray sample_features, np.ndarray outputs,
                    np.ndarray feature_indices, int min_leaf):
    """
    Compute the optimal split across a list of features.

    Arguments:
      sample_features: transposed array of samples
      outputs: array of outputs (float32)
      feature_indices: features to try (int32)
      min_leaf: minimum samples per leaf

    Returns:
      A tuple (feature_idx, split_info) or (None, None).
      split_info is a dict containing the following:
        'threshold': the split threshold.
        'less': sample indices below the threshold.
        'greater': sample indices above the threshold.
        'loss': surrogate loss for the split.
    """
    assert feature_indices.dtype == np.int32
    assert outputs.dtype == np.float32
    if len(outputs) <= min_leaf or len(outputs) < 2:
        return None, None
    if sample_features.dtype == np.uint8:
        return _optimal_feature_uint8(sample_features, outputs,
                                      feature_indices, min_leaf)
    return _optimal_feature_general(sample_features, outputs,
                                    feature_indices, min_leaf)

def _optimal_feature_general(np.ndarray sample_features, np.ndarray outputs,
                             np.ndarray feature_indices, int min_leaf):
    """
    An implementation of optimal_feature() that works with
    any feature type.
    """
    cdef float best_loss = 0
    cdef int[:] raw_indices = feature_indices
    cdef int num_indices = len(feature_indices)
    cdef int i
    cdef int feature_idx

    best_split = None
    cdef int best_feature_idx

    for i in range(num_indices):
        feature_idx = raw_indices[i]
        split = _optimal_split_general(sample_features[feature_idx], outputs,
                                       min_leaf, best_loss)
        if split is not None:
            best_feature_idx = feature_idx
            best_split = split
            best_loss = split['loss']

    if best_split is not None:
        return best_feature_idx, best_split
    return None, None

def _optimal_split_general(np.ndarray sample_features, np.ndarray outputs, int min_leaf,
                           float best_loss):
    """
    Compute the optimal split for any feature dtype.
    """
    indexed_map = zip(sample_features, range(len(outputs)))
    sorted_feats, sorted_indices = zip(*sorted(indexed_map))

    last_feature = sorted_feats[0]
    best_index = None
    best_thresh = None
    losses = _split_losses(outputs[list(sorted_indices)])
    for index, feature in enumerate(sorted_feats):
        if feature > last_feature:
            if index >= min_leaf and index+min_leaf <= len(outputs):
                if losses[index-1] < best_loss:
                    best_loss = losses[index-1]
                    best_thresh = (float(feature) + float(last_feature)) / 2
                    best_index = index
            last_feature = feature
    if best_index is None:
        return None
    return {
        'threshold': best_thresh,
        'less': list(sorted_indices[:best_index]),
        'greater': list(sorted_indices[best_index:]),
        'loss': best_loss
    }

def _split_losses(outputs):
    """
    Compute the loss for every split position.
    """
    left_sums = np.cumsum(outputs, axis=0)[:-1]
    right_sums = np.cumsum(outputs[::-1], axis=0)[::-1][1:]
    left_loss = (np.negative(np.sum(np.square(left_sums), axis=-1)) /
                 np.arange(1, len(left_sums)+1))
    right_loss = (np.negative(np.sum(np.square(right_sums), axis=-1)) /
                  np.arange(len(right_sums), 0, -1))
    return left_loss + right_loss

@cython.boundscheck(False)
def _optimal_feature_uint8(np.ndarray sample_features, np.ndarray outputs,
                           np.ndarray feature_indices, int min_leaf):
    """
    An implementation of optimal_feature() which is
    optimized for uint8 observations.
    """
    cdef int[:] raw_indices = feature_indices
    cdef int num_indices = len(feature_indices)

    cdef int best_feature_idx = -1
    cdef float best_loss = 0
    cdef float best_thresh = 0
    cdef int* best_sorted_indices = NULL
    cdef int best_split_idx = 0

    cdef unsigned char[:, :] raw_features = sample_features
    cdef int num_samples = sample_features.shape[1]
    cdef float[:, :] raw_outputs = outputs
    cdef int num_outputs = outputs.shape[0]
    cdef int output_size = outputs.shape[1]

    cdef int i
    cdef int feature_idx

    with nogil:
        for i in range(num_indices):
            feature_idx = raw_indices[i]
            improved = _optimal_split_uint8(raw_features[feature_idx],
                                            num_samples,
                                            raw_outputs,
                                            num_outputs,
                                            output_size,
                                            min_leaf,
                                            &best_loss,
                                            &best_thresh,
                                            &best_sorted_indices,
                                            &best_split_idx)
            if improved:
                best_feature_idx = feature_idx

    if not best_sorted_indices:
        return None, None

    cdef int[:] py_indices = <int[:num_samples]>best_sorted_indices
    split_info = {
        'threshold': best_thresh,
        'less': list(py_indices[:best_split_idx]),
        'greater': list(py_indices[best_split_idx:]),
        'loss': best_loss
    }
    free(best_sorted_indices)
    return best_feature_idx, split_info

@cython.boundscheck(False)
cdef int _optimal_split_uint8(unsigned char[:] sample_features,
                              int num_samples,
                              float[:, :] outputs,
                              int num_outputs,
                              int output_size,
                              int min_leaf,
                              float* loss_out,
                              float* thresh_out,
                              int** sorted_out,
                              int* split_idx_out) nogil:
    """
    Like _optimal_split_general(), but optimized for uint8
    observations (and without the GIL).

    Arguments:
      sample_features: feature value for each sample
      num_samples: number of entries in sample_features
      outputs: batch of desired outputs
      num_outputs: outer dimension of outputs
      output_size: inner dimension of outputs
      min_leaf: minimum samples per leaf
      loss_out: starts as the current best loss, and is
        changed if a better loss is obtained
      thresh_out: if an improvement is made, this is set
        to the optimal threshold
      sorted_out: if an improvement is made, this is set
        to the sorted sample indices (old value is freed)
      split_idx_out: if an improvement is made, this is
        set to the first index of the greate_equal branch

    Returns:
      1 if improvement was made, 0 otherwise.
    """
    cdef int* sorted_indices

    cdef float* left_sum = <float *>calloc(output_size, sizeof(float))
    cdef float* right_sum = <float *>calloc(output_size, sizeof(float))

    cdef unsigned char last_feature = 0
    cdef unsigned char feature = 0

    cdef float loss
    cdef int i
    cdef float[:] output_vec
    cdef int improved = 0

    sorted_indices = _sort_bytes(sample_features, num_samples)
    _sum_all(right_sum, outputs, output_size, num_outputs)
    for i in range(num_samples):
        feature = sample_features[sorted_indices[i]]
        if i > 0 and feature != last_feature:
            if i >= min_leaf and i+min_leaf <= num_samples:
                loss = _split_loss(left_sum, right_sum, output_size,
                                   i, num_samples)
                if loss < loss_out[0]:
                    thresh_out[0] = (<float>feature + <float>last_feature) / 2
                    loss_out[0] = loss
                    split_idx_out[0] = i
                    improved = 1
        last_feature = feature
        output_vec = outputs[sorted_indices[i]]
        _add_vector(left_sum, output_vec, output_size)
        _sub_vector(right_sum, output_vec, output_size)

    free(left_sum)
    free(right_sum)
    if improved:
        if sorted_out[0]:
            free(sorted_out[0])
        sorted_out[0] = sorted_indices
        return 1
    else:
        free(sorted_indices)
        return 0

@cython.boundscheck(False)
cdef int* _sort_bytes(unsigned char[:] features, int num_samples) nogil:
    """
    Sort the uint8 features and return a sorted list of
    sample indices.

    The caller must free the resulting list
    """
    cdef int feature_counts[0x100]
    cdef int feature_offsets[0x100]
    cdef int accumulator
    cdef unsigned char feature
    cdef int* result = <int *>malloc(num_samples * sizeof(int))
    cdef int i
    for i in range(0x100):
        feature_counts[i] = 0
    for i in range(num_samples):
        feature_counts[features[i]] += 1
    accumulator = 0
    for i in range(0x100):
        feature_offsets[i] = accumulator
        accumulator += feature_counts[i]
    for i in range(num_samples):
        feature = features[i]
        result[feature_offsets[feature]] = i
        feature_offsets[feature] += 1
    return result

@cython.boundscheck(False)
cdef void _add_vector(float* dst, float[:] src, int size) nogil:
    """
    Add the source vector to the destination.
    """
    cdef int i
    for i in range(size):
        dst[i] += src[i]

@cython.boundscheck(False)
cdef void _sub_vector(float* dst, float[:] src, int size) nogil:
    """
    Subtract the source vector from the destination.
    """
    cdef int i
    for i in range(size):
        dst[i] -= src[i]

@cython.boundscheck(False)
cdef void _sum_all(float* dst, float[:,:] src, int size, int count) nogil:
    """
    Add all the source vectors to the destination.
    """
    cdef int i
    for i in range(count):
        _add_vector(dst, src[i], size)

@cython.boundscheck(False)
cdef float _split_loss(float* left, float* right, int size, int i, int count) nogil:
    """
    Compute the loss for a potential split.
    """
    cdef float left_count = <float>i
    cdef float right_count = <float>(count - i)
    return -(_mag_squared(left, size)/left_count +
             _mag_squared(right, size)/right_count)

@cython.boundscheck(False)
cdef float _mag_squared(float* vec, int size) nogil:
    """
    Compute the squared L2 norm.
    """
    cdef int i
    cdef float dot
    for i in range(size):
        dot += vec[i] * vec[i]
    return dot
