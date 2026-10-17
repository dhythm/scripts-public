"""Offline regressions: every secret and filesystem operation uses a temporary fixture."""
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest

SHELL = Path(__file__).resolve().parents[1]


class EnvSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.mock = self.bin / "gcloud"
        self.mock.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
root = pathlib.Path(os.environ["MOCK_ROOT"])
with (root / "calls").open("a") as stream:
    stream.write("called\\n")
values = json.loads((root / "values.json").read_text())
if sys.argv[1:3] == ["secrets", "list"]:
    print("\\n".join(values))
else:
    name = next(arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--secret="))
    if name == os.environ.get("MOCK_FAIL"):
        print("diagnostic must not leak", file=sys.stderr)
        sys.exit(1)
    sys.stdout.write(values[name])
''')
        self.mock.chmod(0o700)
        self.environment = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}",
                                MOCK_ROOT=str(self.root))
        self.set_values({"TOKEN": "dummy"})

    def set_values(self, values):
        (self.root / "values.json").write_text(json.dumps(values))

    def run_script(self, name, *arguments):
        return subprocess.run(["bash", str(SHELL / name), *map(str, arguments)],
                              cwd=self.root, env=self.environment,
                              capture_output=True, text=True)

    def assert_private(self, path):
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_export_quotes_and_preserves_trailing_newlines(self):
        self.set_values({"TOKEN": ' # "quoted" \\path\nline\r\n\n', "EMPTY": ""})
        result = self.run_script("export_gcp_secrets.sh", "project")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / ".env").read_text(),
                         'TOKEN=" # \\"quoted\\" \\\\path\\nline\\r\\n\\n"\nEMPTY=""\n')
        self.assert_private(self.root / ".env")
        self.assertNotIn("quoted", result.stdout + result.stderr)
        self.assertEqual(list(self.root.glob(".gcp-secrets-*")), [])

    def test_existing_export_refused_before_gcloud(self):
        (self.root / ".env").write_text("original")
        (self.root / "secrets_list.txt").write_text("unrelated")
        result = self.run_script("export_gcp_secrets.sh", "project")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "calls").exists())
        self.assertEqual((self.root / ".env").read_text(), "original")
        self.assertEqual((self.root / "secrets_list.txt").read_text(), "unrelated")

    def test_force_export_failure_preserves_old_output_and_cleans_temporary(self):
        (self.root / ".env").write_text("original")
        self.set_values({"TOKEN": "dummy", "BROKEN": "unused"})
        self.environment["MOCK_FAIL"] = "BROKEN"
        result = self.run_script("export_gcp_secrets.sh", "--force", "project")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.root / ".env").read_text(), "original")
        self.assertEqual(list(self.root.glob(".gcp-secrets-*")), [])
        self.assertNotIn("diagnostic must not leak", result.stderr)

    def test_force_export_and_invalid_name(self):
        (self.root / ".env").write_text("original")
        result = self.run_script("export_gcp_secrets.sh", "--force", "project")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_private(self.root / ".env")
        self.set_values({"invalid-name": "dummy"})
        before = (self.root / ".env").read_bytes()
        result = self.run_script("export_gcp_secrets.sh", "--force", "project")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.root / ".env").read_bytes(), before)

    def make_copy_dirs(self):
        source, destination = self.root / "source", self.root / "destination"
        source.mkdir()
        destination.mkdir()
        return source, destination

    def test_copy_conflicts_preflight_then_force_private(self):
        source, destination = self.make_copy_dirs()
        (source / ".env.a").write_text("a")
        (source / ".env.z").write_text("z")
        (destination / ".env.z").write_text("original")
        result = self.run_script("copy-env-files.sh", source, destination)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((destination / ".env.a").exists())
        self.assertEqual((destination / ".env.z").read_text(), "original")
        result = self.run_script("copy-env-files.sh", "--force", source, destination)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((destination / ".env.z").read_text(), "z")
        self.assert_private(destination / ".env.z")

    def test_copy_newline_paths_prunes_nested_destination_and_dependencies(self):
        source, destination = self.make_copy_dirs()
        destination.rmdir()
        destination = source / "worktree"
        destination.mkdir()
        (destination / ".env.existing").write_text("private")
        nested = source / "line\nbreak"
        nested.mkdir()
        (nested / ".env.local").write_text("dummy")
        for name in ("node_modules", ".git"):
            (source / name).mkdir()
            (source / name / ".env").write_text("excluded")
        result = self.run_script("copy-env-files.sh", source, destination)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((destination / "line\nbreak" / ".env.local").read_text(), "dummy")
        self.assertFalse((destination / "worktree").exists())
        self.assertFalse((destination / "node_modules").exists())
        self.assert_private(destination / "line\nbreak" / ".env.local")

    def test_symlink_outputs_and_destination_parents_refused(self):
        outside = self.root / "outside"
        outside.write_text("unchanged")
        (self.root / ".env").symlink_to(outside)
        result = self.run_script("export_gcp_secrets.sh", "--force", "project")
        self.assertNotEqual(result.returncode, 0)
        source, destination = self.make_copy_dirs()
        (source / "sub").mkdir()
        (source / "sub" / ".env").write_text("dummy")
        (destination / "sub").symlink_to(self.root, target_is_directory=True)
        result = self.run_script("copy-env-files.sh", "--force", source, destination)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(outside.read_text(), "unchanged")


if __name__ == "__main__":
    unittest.main()
