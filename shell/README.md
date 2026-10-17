# Environment-file utilities

Both scripts require Bash and Python 3 (standard library only). They refuse to
replace existing destination files unless `--force` is explicitly supplied.
Generated/copied files have permission `0600`, including when replacing an older
file. Neither command prints file contents or secret values.

## Export GCP secrets

```bash
bash shell/export_gcp_secrets.sh PROJECT_ID [OUTPUT_FILE]
bash shell/export_gcp_secrets.sh --force PROJECT_ID .env
```

The default output is `.env` in the current directory. The output directory must
already exist. `gcloud` must be installed and authenticated for the project.
The command lists secrets and accesses their latest versions; run only against a
project you intend to export. Existing output is rejected before contacting GCP.
A failed export leaves the previous output intact. Successful exports publish a
complete file atomically; no fixed `secrets_list.txt` is created or removed.
Symbolic-link output files are rejected, even with `--force`.

Values use double-quoted dotenv syntax with escaped backslashes, quotes, LF and
CR. Empty values and trailing newlines are preserved. Names must match
`[A-Za-z_][A-Za-z0-9_]*`; invalid names and NUL-containing values abort the export.
Use a dotenv parser that supports these escapes. Disable interpolation when
loading arbitrary secret values (for example, Python dotenv's
`load_dotenv(interpolate=False)`); do not source this file as shell code.

## Copy environment files

```bash
bash shell/copy-env-files.sh SOURCE_DIR DEST_DIR
bash shell/copy-env-files.sh --force SOURCE_DIR DEST_DIR
```

Both directories must already exist and differ. The command copies regular
`.env*` files recursively, preserving relative paths. It skips `.git`,
`node_modules`, source symlinks, and the destination subtree when it is inside
the source. Destination symlinks are rejected. Paths containing spaces or
newlines work without line-based filename parsing.

Known destination conflicts are checked before any file is copied. Each file
is published atomically; the operation is not a transaction across all files,
so an I/O error during copying can leave earlier files successfully copied.
Run in directories you control; this is not a defense against another process
concurrently replacing directory components.

## Offline regression tests

```bash
bash -n shell/export_gcp_secrets.sh shell/copy-env-files.sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s shell/tests -v
```

The tests use only dummy values, a mock `gcloud` executable, and temporary
directories. They never contact GCP or inspect real environment files.
