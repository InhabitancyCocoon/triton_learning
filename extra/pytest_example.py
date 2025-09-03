import os
import pytest

def test_foo():
    print(os.environ.get("PYTEST_VERSION"))
    assert 1 == 1