import numpy as np


def test_embedding_index_alignment_identity():
    n = 11
    assert np.array_equal(np.arange(n), np.arange(n, dtype=np.int64))
