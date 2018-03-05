from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='pandas_string',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    description='String type for pandas (based on Apache Arrow)',
    url='https://github.com/xhochy/pandas-string',
    author='Uwe L. Korn',
    author_email='pandas@uwekorn.com',
    license="MIT",
    classifiers=[  # Optional
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(),
    install_requires=[
        'pandas>=0.23.0.dev0',
    ]
)
