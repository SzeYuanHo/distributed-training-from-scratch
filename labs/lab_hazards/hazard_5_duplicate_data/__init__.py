"""Makes this hazard a package so pytest imports its `broken` module distinctly.

All five hazards ship a file called `broken.py`. Without these package markers,
pytest would import the first one it finds and reuse it for every hazard, and
four of the five test files would silently test the wrong code.
"""
