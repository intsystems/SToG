"""Sphinx configuration for SToG documentation."""

import os
import sys
from pathlib import Path

# ============================================================================
# Path Setup
# ============================================================================

# Get absolute path to project root
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root / "src"))

print(f"Python path: {sys.path[0]}")
print(f"Project root: {project_root}")

# ============================================================================
# Project Information
# ============================================================================

project = 'SToG'
copyright = '2025, MIPT Intelligent Systems'
author = 'Eynullayev A., Rubtsov D., Firsov S., Karpeev G.'
release = '0.0.1'
version = '0.0.1'

# ============================================================================
# General Configuration
# ============================================================================

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'myst_parser',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

master_doc = 'index'
language = 'en'
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# ============================================================================
# Autodoc Configuration
# ============================================================================

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': False,
    'show-inheritance': True,
}

autosummary_generate = True
autosummary_generate_overwrite = True

# ============================================================================
# Napoleon Configuration (Google docstrings)
# ============================================================================

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_param = True
napoleon_use_rtype = True

# ============================================================================
# HTML Theme Configuration
# ============================================================================

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'style_nav_header_background': '#3f51b5',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
}

html_static_path = ['_static']
templates_path = ['_templates']

# Create empty _static and _templates if they don't exist
_static_path = Path(__file__).parent / '_static'
_templates_path = Path(__file__).parent / '_templates'
_static_path.mkdir(exist_ok=True)
_templates_path.mkdir(exist_ok=True)

# ============================================================================
# Intersphinx Configuration
# ============================================================================

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
}

# ============================================================================
# Suppress Warnings
# ============================================================================

suppress_warnings = [
    'app.add_config_value',
]
