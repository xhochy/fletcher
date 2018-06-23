from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name="fletcher",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description="Pandas ExtensionDType/Array backed by Apache Arrow",
    url="https://github.com/xhochy/fletcher",
    author="Uwe L. Korn",
    author_email="fletcher@uwekorn.com",
    license="MIT",
    classifiers=[  # Optional
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    packages=find_packages(),
    install_requires=["pandas>=0.23.0", "pyarrow>=0.9.0", "numba", "six"],
)
