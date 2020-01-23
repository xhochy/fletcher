fletcher
========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   API Reference <api/modules>
   Motivation <motivation>

Use Apache Arrow backed columns in Pandas 0.23+ using the ExtensionArray interface.

Fletcher provides a generic implementation of the ``ExtensionDtype`` and
``ExtensionArray`` interfaces of Pandas for columns backed by Apache Arrow. By
using it you can use any data type available in Apache Arrow natively in Pandas.
Most prominently, ``fletcher`` provides native String und List types.

``fletcher`` provides two, slightly different implementations. There is
:class:`~fletcher.FletcherChunkedArray` which is based on
``pyarrow.ChunkedArray``, i.e. it consists of a collection of one or more
continuous ``pyarrow.Array`` instances. Thus the backing memory can be a
single memory region but it isn't required. This makes operations like
``concat`` copy-free as the result will be a ``ChunkedArray`` that consists
of the union of the chunks of the inputs. In contrast it makes algorithm
implementation a bit more complex as we need to implement all algorithms to
iterate over all rows of all the arrays, not simply 0..n-1 of a single array.

The other implementation is :class:`~fletcher.FletcherContinuousArray`
which is based on a single ``pyarrow.Array`` instance. While this makes
operations like ``concat`` more costly, it greatly improves usability and
extensibility by being a much simpler structure. One can always assume that
the backing memory region is a continuous block of memory and iterate with
simple 0..n-1 indexing over the rows.

At the moment, we don't provide a default ``FletcherArray``-named
implementation as we are uncertain which of the two above implementations will
be the most accepted one. Once we know to which implementation users converge,
we will name that one ``FletcherArray``.

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


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
