#!/usr/bin/env python3
"""Verify the portable bundle without importing ML libraries or using a GPU."""
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "BUNDLE_MANIFEST.json").read_text())
    for row in manifest["files"]:
        path = ROOT / row["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise SystemExit(f"Missing/changed bundled file: {row['path']}")
    count = 0
    for folder in ("st-ghost", "GHOST-master", "tools", "tests"):
        for path in (ROOT / folder).rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path))
            count += 1
    for size, per_pair in ((2000, 400), (1000, 200)):
        rows = json.loads((ROOT / f"migration_inputs/historical/selected_images_{size}.json").read_text())
        counts = Counter(r["spec"] for r in rows)
        assert len(rows) == size and len({r["image_id"] for r in rows}) == size
        assert len(counts) == 5 and set(counts.values()) == {per_pair}
    print(f"OK: {len(manifest['files'])} file checksums, {count} Python files, historical 2000/1000 image manifests.")
    print("Model weights, COCO data, trained projector and GPU runtime are not checked by this command.")


if __name__ == "__main__":
    main()
