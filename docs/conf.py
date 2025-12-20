import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

project = "lightweight-mcnnm"
copyright = "2024, Tobias Schnabel"
author = "Tobias Schnabel"
release = "1.1.3"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    # "sphinx_gallery.gen_gallery",
]

extensions.append("sphinx_autodoc_typehints")
extensions.append("sphinx_tabs.tabs")


# sphinx_gallery_conf = {
#     'examples_dirs': '../examples',
#     'gallery_dirs': 'auto_examples',
#     'filename_pattern': '/example_',
#     'ignore_pattern': r'__init__\.py',
# }


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Use the Sphinx Book Theme
html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/tobias-schnabel/mcnnm",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "use_fullscreen_button": False,
    "navbar_end": ["navbar-icon-links.html", "search-field.html"],
    "search_bar_text": "Search the docs...",
}
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# Configurations for sphinx_autodoc_typehints
set_type_checking_flag = True
typehints_fully_qualified = False
always_document_param_types = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Enable LaTeX rendering
mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    },
}

# Add any Sphinx extension module names here, as strings
extensions.append("sphinx_book_theme")
