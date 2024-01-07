"""
H2O AutoML Widget
===============

Widgets for running H2O AutoML
"""
import sysconfig

NAME = "H2O AutoML"
DESCRIPTION = "AutoML Widgets"

ICON = "icons/h2o-logo.svg"
PRIORITY = 1000
BACKGROUND = "#ffe600"

WIDGET_HELP_PATH = (
# Used for development.
# You still need to build help pages using
# make html
# inside doc folder
("{DEVELOP_ROOT}/doc/_build/html/index.html", None),

# Documentation included in wheel
# Correct DATA_FILES entry is needed in setup.py and documentation has to be
# built before the wheel is created.
("{}/help/orange3-automl/index.html".format(sysconfig.get_path("data")),
 None),

# Online documentation url, used when the local documentation is available.
# Url should point to a page with a section Widgets. This section should
# includes links to documentation pages of each widget. Matching is
# performed by comparing link caption to widget name.
("http://orange3-automl.readthedocs.io/en/latest/", "")
)
