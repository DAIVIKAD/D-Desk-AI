"""
Small CLI wrapper used by the FastAPI app to run CNN image inference inside
the TensorFlow environment (venv_tf).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.ml.image_model import classify_image


def main() -> int:
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python -m app.ml.image_infer <image_path>"}))
        return 2

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(json.dumps({"error": f"Image file not found: {image_path}"}))
        return 2

    result = classify_image(image_path.read_bytes())
    print(json.dumps(dict(result)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
