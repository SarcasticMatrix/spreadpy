# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'spreadpy'
copyright = '2025, Théophile Schmutz'
author = 'Théophile Schmutz'
release = 'v0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_design",
    "myst_parser",
    'nbsphinx',
    'sphinx.ext.githubpages',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]
templates_path = ['_templates']
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store',
    'api/models.rst', 'api/calibration.rst',
    'examples/pricing', 'examples/calibration',
    'examples/hedging', 'examples/asset_allocation',
    'examples/asset_allocations.rst',
]
highlight_language = 'python'

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]

html_theme_options = {
   "logo": {
      "image_light": "_static/logo-light-no-bg.png",
      "image_dark": "_static/logo-light-no-bg.png",
   },

    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/SarcasticMatrix/spreadpy",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],

    "show_toc_level": 3,
    "show_nav_level": 2,
    "navigation_depth": 2,

    "pygments_light_style": "xcode",
    "pygments_dark_style": "lightbulb"
}
