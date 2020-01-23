Motivation
==========

Fletcher was started as weekend project to see how far you could get by using pandas' new ``ExtensionArray`` interface together with Numba and Apache Arrow to build a fast string type. Since then it has evolved into a project to explore the general issues that you experience when using a non-numpy storage for pandas columns.

Fletchers main aim is now to provide a general ``ExtensionArray`` implementation to support Apache Arrow-backed columns in ``pandas``. We restrict ourselves in the development of fletcher to use only pure Python code and do all compilation just-in-time using Numba. On the one side, this makes distribution of this package much simpler as we don't ship native code, on the other side it is also dogfooding for the Apache Arrow developers working on this project to better understand what is needed to accelerate Python code using Numba on top of Apache Arrow.

While Fletcher is currently not in a state where you can use it in day-to-day work, it already provides valuable feedback to pandas on where its ExtensionArray interface is still bound to NumPy. In addition, it also provides feedback to Apache Arrow on what functionality is missing to use it as the backend of a DataFrame library. A long-term goal of fletcher is to provide enough input to the pandas and Apache Arrow community to eventually let pandas use ``pyarrow`` as a hard dependency for backing some of its column types.

As an end-user, you watch Fletcher's development over the next months when you are interested in the efficient implementation of the following data types:

 * Nullability for numerical data (all other data supported by fletcher also supports efficient nullability masks)
 * strings
 * nested structures like ``List[…]`` or ``Struct[…]`` or any recursive combination of that.
