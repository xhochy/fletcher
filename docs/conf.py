# -*- coding: utf-8 -*-

project = "fletcher"
copyright = "2018-2020, Fletcher contributors"
author = "Fletcher contributors"

extensions = ["numpydoc", "sphinxcontrib.apidoc"]

apidoc_module_dir = "../fletcher"
apidoc_output_dir = "api"
apidoc_separate_modules = True
apidoc_extra_args = ["--implicit-namespaces"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
