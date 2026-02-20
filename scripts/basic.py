"""Basic example: import streamcutter and verify a function works.

Install the package first:
    pip install -e .

Then run this script:
    python scripts/basic.py
"""

import streamcutter

version = streamcutter.get_version()
print(f"streamcutter version: {version}")
assert version == streamcutter.__version__, "get_version() mismatch"
print("Basic check passed.")
