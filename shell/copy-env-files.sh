#!/bin/bash
# Usage: ./copy-env-files.sh [--force] SOURCE_DIR DEST_DIR
# Requires python3. Existing destination files are protected by default.
set -euo pipefail
exec python3 - "$@" <<'PYTHON'
import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile

parser = argparse.ArgumentParser(description="Copy .env* files without exposing or clobbering them.")
parser.add_argument("--force", action="store_true", help="atomically replace existing files")
parser.add_argument("source")
parser.add_argument("destination")
args = parser.parse_args()
try:
    source = Path(args.source).resolve(strict=True)
    destination = Path(args.destination).resolve(strict=True)
    if not source.is_dir() or not destination.is_dir():
        raise ValueError("Source and destination must be directories")
    if source == destination:
        raise ValueError("Source and destination must differ")
    copies = []
    for root, directories, files in os.walk(source, followlinks=False):
        root = Path(root)
        directories[:] = [name for name in directories
                          if name not in ("node_modules", ".git")
                          and not (root / name).is_symlink()
                          and (root / name).resolve() != destination]
        for name in sorted(files):
            src = root / name
            if not name.startswith(".env") or src.is_symlink() or not src.is_file():
                continue
            target = destination / src.relative_to(source)
            copies.append((src, target))

    def validate(target):
        # Do not follow destination symlinks into unrelated directories or files.
        for part in [target, *target.parents]:
            if part == destination:
                break
            if part.is_symlink():
                raise ValueError("Refusing a symbolic-link destination")
        if target.exists() and (not args.force or not target.is_file()):
            raise ValueError("Destination exists; use --force to replace regular files")

    # Find known conflicts before copying anything.
    for _, target in copies:
        validate(target)
    for src, target in copies:
        validate(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".env-copy-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream, src.open("rb") as incoming:
                shutil.copyfileobj(incoming, stream)
                stream.flush()
                os.fsync(stream.fileno())
            validate(target)
            if args.force:
                os.replace(temporary, target)
            else:
                os.link(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    print(f"Copied {len(copies)} environment files")
except (OSError, ValueError) as error:
    print(f"Error: {error}", file=sys.stderr)
    sys.exit(1)
PYTHON
