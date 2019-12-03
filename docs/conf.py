# -*- coding: utf-8 -*-

project = "fletcher"
copyright = "2018, Florian Jetter, Christopher Prohm, Uwe Korn"
author = "Florian Jetter, Christopher Prohm, Uwe Korn"

extensions = ["numpydoc", "sphinxcontrib.apidoc"]

# extensions = [
#    "sphinx.ext.autodoc",
#    "sphinx.ext.doctest",
#    "sphinx.ext.intersphinx",
#    "sphinx.ext.coverage",
#    "sphinx.ext.viewcode",
#    "sphinx.ext.githubpages",
# ]

extensions = ["numpydoc", "sphinxcontrib.apidoc"]

apidoc_module_dir = "../fletcher"
apidoc_output_dir = "api"
apidoc_separate_modules = True
apidoc_extra_args = ["--implicit-namespaces"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
