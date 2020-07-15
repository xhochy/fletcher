from typing import Any, List

from hypothesis.core import Example


def examples(example_list: List[Any], example_kword: str):
    """
    Annotation for tests using hypothesis input generation.

    It is similar to the @example annotation but allows specifying a list of examples.

    Parameters
    ----------
    example_list:
        list of examples
    example_kword:
        which parameter to use for passing the example values to the test function

    Returns
    -------
    method:
        wrapper method
    """

    def accept(test):
        if not hasattr(test, "hypothesis_explicit_examples"):
            test.hypothesis_explicit_examples = []
        test.hypothesis_explicit_examples.extend(
            [Example((), {example_kword: ex}) for ex in example_list]
        )
        return test

    return accept
