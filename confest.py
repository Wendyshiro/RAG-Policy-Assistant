import sys
from pathlib import Path

# Add the project root to sys.path so 'src' is importable in all tests
sys.path.insert(0, str(Path(__file__).resolve().parent))