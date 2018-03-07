import operator as op

import pandas as pd
import pandas.util.testing as pdt
import pytest

import pandas_string as pd_str


data = ['foo', None, 'baz', 'bar', None, '..bar']

df = pd.DataFrame({
    'pd': pd.Series(data),
    'pd_str': pd_str.StringArray(data)
})


test_cases = [
    dict(label='startswith', method='startswith', args=['ba']),
    dict(label='endswith', method='endswith', args=['ar']),
]


@pytest.mark.parametrize('spec', test_cases, ids=op.itemgetter('label'))
def test_reference_impl(spec):
    expected = getattr(df['pd'].str, spec['method'])(*spec.get('args', []), *spec.get('kwargs', {}))
    actual = getattr(df['pd_str'].text, spec['method'])(*spec.get('args', []), *spec.get('kwargs', {}))

    pdt.assert_series_equal(expected, actual, check_names=False)
