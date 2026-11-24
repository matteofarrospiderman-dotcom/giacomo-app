import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# isolate test data from real projects
os.environ.setdefault("TABFORGE_DATA_DIR", tempfile.mkdtemp(prefix="tabforge-test-"))
os.environ.setdefault("TABFORGE_SEPARATOR", "dsp")
