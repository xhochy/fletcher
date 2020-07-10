Frequently Asked Questions
==========================

Roadmap
-------

Should this be merged into ``pandas`` one day?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, we definitely want to have parts of ``fletcher`` as part of ``pandas``.
There are `plans for a native string type <https://github.com/pandas-dev/pandas/issues/35169>`_ and a `list type <https://github.com/pandas-dev/pandas/issues/35176>`_ where Apache Arrow would be the preferable data structure.

If this is merged into ``pandas``, is there still a need for ``fletcher``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Definitely! ``fletcher`` functions as glue for all available types in Apache Arrow as ``pandas.ExtensionDtype``.
Only a subset of these types will make its way into ``pandas``, thus the string functions that return a boolean result in ``pandas`` will return a ``pandas.BooleanArray`` (bytemask-backed ``numpy`` array) whereas the ones in ``fletcher`` will keep returning a :class:`fletcher.FletcherBaseArray` (bitmask-backed Arrow array).

Furthermore, an additional goal of ``fletcher`` is to support working with ``numba`` on top of ``pyarrow.Array`` structures.
These utilities will not make their way into ``pandas`` as there we don't wont to introduce a hard ``numba`` dependency.

Development
-----------

