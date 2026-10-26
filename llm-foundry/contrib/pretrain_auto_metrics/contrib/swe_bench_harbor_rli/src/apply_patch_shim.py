#!/usr/bin/env python3
"""OpenAI-style ``apply_patch`` shim for environments without one installed.

gpt-oss (and other Codex-trained models) emit edits in OpenAI's
``apply_patch`` envelope, but the RLI base image does not ship the tool, so
every such call returns ``bash: apply_patch: command not found`` (rc=127) and
the agent's step budget is burned. This shim is installed by
``RLIEnvironment.start()`` into ``/usr/local/bin/apply_patch`` so the model's
native edit format works.

Patch format::

    *** Begin Patch
    *** Update File: <path>
    @@ optional section header
     context line
    -removed line
    +added line
    *** Add File: <path>
    +new line
    *** Delete File: <path>
    *** End Patch

Reads the patch from ``argv[1]`` if given, otherwise stdin. Exits 0 on
success; non-zero with a short diagnostic on parse/apply failure.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _read_patch() -> str:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).read_text()
    return sys.stdin.read()


def _parse(text: str) -> list[dict]:
    lines = text.splitlines()
    i = 0
    while i < len(lines) and not lines[i].startswith("*** Begin Patch"):
        i += 1
    if i >= len(lines):
        sys.exit("apply_patch: missing '*** Begin Patch'")
    i += 1
    actions: list[dict] = []
    current: dict | None = None
    while i < len(lines):
        line = lines[i]
        if line.startswith("*** End Patch"):
            if current is not None:
                actions.append(current)
                current = None
            i += 1
            break
        if line.startswith("*** Update File:"):
            if current is not None:
                actions.append(current)
            current = {
                "op": "update",
                "path": line[len("*** Update File:"):].strip(),
                "hunks": [[]],
            }
        elif line.startswith("*** Add File:"):
            if current is not None:
                actions.append(current)
            current = {
                "op": "add",
                "path": line[len("*** Add File:"):].strip(),
                "lines": [],
            }
        elif line.startswith("*** Delete File:"):
            if current is not None:
                actions.append(current)
            current = {
                "op": "delete",
                "path": line[len("*** Delete File:"):].strip(),
            }
        elif current is None:
            pass  # stray noise before first directive
        elif current["op"] == "update":
            if line.startswith("@@"):
                current["hunks"].append([])
            else:
                current["hunks"][-1].append(line)
        elif current["op"] == "add":
            if line.startswith("+"):
                current["lines"].append(line[1:])
            else:
                current["lines"].append(line)
        i += 1
    if current is not None:
        actions.append(current)
    return actions


def _apply_update(path: str, hunks: list[list[str]]) -> None:
    p = Path(path)
    if not p.exists():
        sys.exit(f"apply_patch: file not found: {path}")
    src = p.read_text()
    file_lines = src.splitlines()
    trailing_newline = src.endswith("\n")

    for hunk_lines in hunks:
        if not hunk_lines:
            continue
        expected: list[str] = []
        replacement: list[str] = []
        for raw in hunk_lines:
            if raw == "":
                expected.append("")
                replacement.append("")
                continue
            tag = raw[0]
            rest = raw[1:] if tag in " +-" else raw
            if tag == "+":
                replacement.append(rest)
            elif tag == "-":
                expected.append(rest)
            else:
                expected.append(rest)
                replacement.append(rest)

        if not expected:
            # Pure insertion with no anchor: append at end of file
            file_lines.extend(replacement)
            continue

        n = len(expected)
        idx = -1
        for j in range(len(file_lines) - n + 1):
            if file_lines[j:j + n] == expected:
                idx = j
                break
        if idx == -1:
            stripped = [s.rstrip() for s in expected]
            for j in range(len(file_lines) - n + 1):
                if [s.rstrip() for s in file_lines[j:j + n]] == stripped:
                    idx = j
                    break
        if idx == -1:
            sys.exit(f"apply_patch: hunk not found in {path}")
        file_lines = file_lines[:idx] + replacement + file_lines[idx + n:]

    out = "\n".join(file_lines)
    if trailing_newline:
        out += "\n"
    p.write_text(out)


def _apply_add(path: str, lines: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(lines)
    if lines:
        body += "\n"
    p.write_text(body)


def _apply_delete(path: str) -> None:
    p = Path(path)
    if p.exists():
        p.unlink()


def main() -> None:
    actions = _parse(_read_patch())
    for action in actions:
        op = action["op"]
        if op == "update":
            _apply_update(action["path"], action["hunks"])
        elif op == "add":
            _apply_add(action["path"], action["lines"])
        elif op == "delete":
            _apply_delete(action["path"])
    print("Done")


if __name__ == "__main__":
    main()
