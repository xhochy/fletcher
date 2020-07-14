import pytest
import fletcher as fr


@pytest.mark.parametrize(
    ['data', 'slice_', 'expected'],
    [
        (['abcd', 'defg', 'hijk'], (1, 2, 1), ['b', 'e', 'i']),
        (['abcd', 'defg', 'h'], (1, 2, 1), ['b', 'e', '']),
        (['abcd', 'defg', 'hijk'], (1, 4, 2), ['bd', 'eg', 'ik']),
        (['abcd', 'defg', 'hijk'], (0, -2, 1), ['ab', 'de', 'hi']),
        ([''])
    ]
)
def test_slice(data, slice_, expected):
    """
    - test getting the right string
    - test null handling
    - test negative slice
    - test step
    - test negative step
    - test unicode
    - test outside bound -> empty string
    """