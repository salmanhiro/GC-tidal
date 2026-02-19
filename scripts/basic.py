"""Basic example: import pyfalconstream and verify a function works.

Install the package first:
    pip install -e .

Then run this script:
    python scripts/basic.py
"""

import pyfalconstream

version = pyfalconstream.get_version()
print(f"pyfalconstream version: {version}")
assert version == pyfalconstream.__version__, "get_version() mismatch"
print("Basic check passed.")
