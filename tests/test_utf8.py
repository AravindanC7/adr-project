"""Text files are always read and written as UTF-8, whatever the machine's default.

On Windows the default text encoding is usually CP-1252, which can't hold characters like the
ranking arrows ("↑") the browser agent reads off the page, so a finished run crashed while
saving its audit files. Two guards:

1. Every text open()/read_text()/write_text() in adr/ and scripts/ passes encoding=.
2. A copy-paste run and a document-upload run, in a fresh Python started with
   -X warn_default_encoding, with EncodingWarning from our code turned into an error, write and
   read back answers, errors and a config note containing "↑ é 中".
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = "↑ é 中"
TEXT_IO_METHODS = {"open", "read_text", "write_text"}


def _mode(call: ast.Call) -> str:
    """The mode argument of open(path, mode) or Path.open(mode), if it's a literal."""
    is_builtin = isinstance(call.func, ast.Name)
    pos = 1 if is_builtin else 0
    if len(call.args) > pos and isinstance(call.args[pos], ast.Constant):
        return str(call.args[pos].value)
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            return str(kw.value.value)
    return "r"


def _text_io_without_encoding(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
        if name not in TEXT_IO_METHODS or (name == "open" and "b" in _mode(node)):
            continue
        if (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and f.value.id in {"webbrowser", "page", "context", "browser"}
        ):
            continue
        if not any(kw.arg == "encoding" for kw in node.keywords):
            found.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    return found


def test_every_text_file_access_names_its_encoding():
    files = sorted([*(ROOT / "adr").rglob("*.py"), *(ROOT / "scripts").rglob("*.py")])
    missing = [hit for p in files for hit in _text_io_without_encoding(p)]
    assert missing == [], "pass encoding='utf-8' at: " + ", ".join(missing)


RUN_BOTH = textwrap.dedent(f"""
    import json, re, sys, warnings
    from pathlib import Path
    warnings.filterwarnings("error", category=EncodingWarning, module=r"(adr|scripts|tests)(\\.|$)")
    import yaml
    from adr.harness.copy_paste import CopyPasteConfig, run_copy_paste
    from adr.harness.doc_upload import DocUploadConfig, run_doc_upload
    from tests import test_doc_upload as du, test_harness as cp

    root = Path(sys.argv[1])
    sample = {SAMPLE!r}

    def cp_model(system, messages):           # every reply carries the sample text
        return cp.naive_model(system, messages) + " " + sample

    def du_model(system, messages, attachments=None):
        return du.reads_pdf(system, messages, attachments).replace("coworkers.", "coworkers " + sample)

    for _ in range(2):                         # the second pass reads every answer from the cache
        client, _f = cp.make_client(root / "cp", fn=cp_model)
        a, items, variants, mechs = cp._setup()
        run_copy_paste(client, a, items, variants, mechs,
                       CopyPasteConfig(run_id="cp", models=["fake-a"], note=sample,
                                       results_dir=str(root)))
        client, _f = du.make_client(root / "du", fn=du_model)
        a, items, variants, mechs = du._setup()
        run_doc_upload(client, a, items, variants, mechs,
                       DocUploadConfig(run_id="du", models=["fake-a"], note=sample,
                                       results_dir=str(root)))
""")


def test_runs_write_and_read_non_ascii_text_as_utf8(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-X", "warn_default_encoding", "-c", RUN_BOTH, str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    for run in ("cp", "du"):
        run_dir = tmp_path / "runs" / run
        config = (run_dir / "config.yaml").read_text(encoding="utf-8")
        assert SAMPLE in config  # readable, not \\u-escaped
        responses = (run_dir / "responses.jsonl").read_text(encoding="utf-8")
        errors = (run_dir / "errors.jsonl").read_text(encoding="utf-8")
        assert SAMPLE in responses + errors
