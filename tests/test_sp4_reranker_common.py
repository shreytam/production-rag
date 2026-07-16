import pytest
from providers.rerankers._common import min_max_normalize

def test_min_max_normalize():
    # Test typical scores
    assert min_max_normalize([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]

    # Test identical scores (division by zero hazard)
    assert min_max_normalize([2.5, 2.5, 2.5]) == [1.0, 1.0, 1.0]

    # Test empty list (should gracefully return empty list)
    assert min_max_normalize([]) == []
