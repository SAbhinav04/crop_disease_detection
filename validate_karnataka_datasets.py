#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    ml_script = Path(__file__).parent / "ml_pipeline" / "validate_karnataka_datasets.py"
    # Pass --project-root so it points to the actual repository root
    args = sys.argv[1:]
    if "--project-root" not in args:
        args.extend(["--project-root", str(Path(__file__).parent.resolve())])
    sys.exit(subprocess.call([sys.executable, str(ml_script)] + args))
