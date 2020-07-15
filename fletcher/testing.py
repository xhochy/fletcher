from hypothesis.core import Example


def examples(example_list, example_kword):
    def accept(test):
        if not hasattr(test, "hypothesis_explicit_examples"):
            test.hypothesis_explicit_examples = []
        test.hypothesis_explicit_examples.extend(
            [Example((), {example_kword: ex}) for ex in example_list]
        )
        return test

    return accept
