import glob
import os.path

from flake8.api import legacy as flake8


def test_codestyle():
    basedir = os.path.dirname(__file__)

    style_guide = flake8.get_style_guide(max_line_length=88)
    report = style_guide.check_files(
        glob.glob(os.path.abspath(os.path.join(basedir, "*.py")))
        + glob.glob(os.path.abspath(os.path.join(basedir, "..", "fletcher", "*.py")))
        + glob.glob(os.path.abspath(os.path.join(basedir, "..", "benchmarks", "*.py")))
    )

    assert report.total_errors == 0
