"""Point d'entrée : exécute tout le pipeline de bout en bout."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from churn.pipeline import run  # noqa: E402

if __name__ == "__main__":
    run()
