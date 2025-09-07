import numpy as np
from numpy.testing import assert_equal
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from clamp.utils import resp_matrix


@given(
    cluster_ids=arrays(
        dtype=np.intp,
        elements=st.integers(min_value=0, max_value=1024),
        shape=st.integers(min_value=1, max_value=2048),
    ),
)
def test_resp_matrix(cluster_ids: np.ndarray[tuple[int], np.dtype[np.integer]]):
    resp = resp_matrix(cluster_ids)
    for c, r in zip(cluster_ids, resp, strict=True):
        expected = np.zeros_like(r)
        expected[c] = 1
        assert_equal(r, expected)
