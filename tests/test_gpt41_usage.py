from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".ipynb"}
MODEL_PATTERN = re.compile(r"openai/gpt-[a-z0-9.\-]+|(?<![a-z0-9_/\.\-])gpt-[0-9][a-z0-9.\-]*")
ALLOWED = {"openai/gpt-4.1", "gpt-4.1"}


def iter_text_files():
    for path in ROOT.rglob("*"):
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if "__pycache__" in path.parts:
            continue
        yield path


def collect_model_matches():
    findings = {}
    for path in iter_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        matches = MODEL_PATTERN.findall(text.lower())
        if matches:
            findings[path.relative_to(ROOT).as_posix()] = sorted(set(matches))
    return findings


class GPT41UsageTests(unittest.TestCase):
    def test_only_expected_gpt_model_strings_exist(self):
        findings = collect_model_matches()
        bad = {
            path: matches
            for path, matches in findings.items()
            if any(match not in ALLOWED for match in matches)
        }
        self.assertEqual({}, bad)


if __name__ == "__main__":
    unittest.main()
