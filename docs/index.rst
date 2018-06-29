fletcher
========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Use Apache Arrow backed columns in Pandas 0.23+ using the ExtensionArray interface.

Fletcher provides a generic implementation of the ``ExtensionDtype`` and
``ExtensionArray`` interfaces of Pandas for columns backed by Apache Arrow. By
using it you can use any data type available in Apache Arrow natively in Pandas.
Most prominently, ``fletcher`` provides native String und List types.

In addition to bringing an alternative memory backend to NumPy, ``fletcher``
also provides high-performance operations on the new column types. It will
either use the native implementation of an algorithm if provided in ``pyarrow``
or otherwise provide an implementation by itself using Numba.

Usage of fletcher columns is straightforward using Pandas' default constructor:

.. code::

    import fletcher as fr
    import pandas as pd

    df = pd.DataFrame({
        'str_column': fr.FletcherArray(['Test', None, 'Strings'])
    })
    df.info()

    # <class 'pandas.core.frame.DataFrame'>
    # RangeIndex: 3 entries, 0 to 2
    # Data columns (total 1 columns):
    # str_column    2 non-null string
    # dtypes: string(1)
    # memory usage: 108.0 bytes


* :ref:`genindex`
* :ref:`modindex`
