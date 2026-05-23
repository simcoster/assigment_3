"""Download and cache the Bitext dataset."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import download_dataset


def main() -> None:
    path = download_dataset()
    print(f"Dataset cached at: {path}")


if __name__ == "__main__":
    main()
