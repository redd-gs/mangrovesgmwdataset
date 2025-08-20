# Ensure the project root and src are on sys.path for tests
import sys
from pathlib import Path
root = Path(__file__).parent.resolve()
src = root / 'src'
for p in (root, src):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
