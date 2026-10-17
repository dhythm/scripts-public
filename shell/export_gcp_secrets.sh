#!/bin/bash
# Usage: ./export_gcp_secrets.sh [--force] PROJECT_ID [OUTPUT_FILE]
# Requires python3 and gcloud. Existing output is preserved unless --force is used.
set -euo pipefail
exec python3 - "$@" <<'PYTHON'
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser(description="Export GCP secrets to a private dotenv file.")
parser.add_argument("--force", action="store_true", help="atomically replace an existing output")
parser.add_argument("project_id")
parser.add_argument("output", nargs="?", default=".env")
args = parser.parse_args()
output = Path(args.output).absolute()
temporary = None
try:
    if output.is_symlink():
        raise ValueError("Refusing a symbolic-link output")
    if output.exists() and not args.force:
        raise ValueError("Output exists; use --force to replace it")
    # Capture errors too: do not let command diagnostics accidentally disclose values.
    def gcloud(*arguments):
        result = subprocess.run(["gcloud", *arguments], capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError("gcloud failed; output was not replaced")
        return result.stdout.decode("utf-8")

    names = gcloud("secrets", "list", "--project=" + args.project_id,
                   "--format=value(name)").splitlines()
    for name in names:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError("A secret name is not a valid dotenv variable name")
    descriptor, temporary = tempfile.mkstemp(prefix=".gcp-secrets-", dir=output.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        for name in names:
            value = gcloud("secrets", "versions", "access", "latest",
                           "--secret=" + name, "--project=" + args.project_id)
            if "\x00" in value:
                raise ValueError("A secret contains a NUL byte and cannot be stored in dotenv")
            # Double-quoted dotenv strings preserve #, whitespace and trailing newlines.
            value = (value.replace("\\", "\\\\").replace('"', '\\"')
                     .replace("\n", "\\n").replace("\r", "\\r"))
            stream.write(f'{name}="{value}"\n')
        stream.flush()
        os.fsync(stream.fileno())
    # mkstemp creates mode 0600; never expose a partially written output.
    if args.force:
        if output.is_symlink():
            raise ValueError("Refusing a symbolic-link output")
        os.replace(temporary, output)
    else:
        # Atomic no-clobber publication, even if another process creates output now.
        os.link(temporary, output)
    print(f"Exported {len(names)} secrets to {output}")
except (OSError, ValueError, RuntimeError) as error:
    print(f"Error: {error}", file=sys.stderr)
    sys.exit(1)
finally:
    if temporary is not None:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
PYTHON
