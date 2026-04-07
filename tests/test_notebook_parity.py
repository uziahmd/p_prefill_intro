from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rq1"))

from prefill_workflows.parity import audit_notebook_parity


class NotebookParityTests(unittest.TestCase):
    def test_notebook_parity_audit_has_no_mismatches(self):
        mismatches = [check for check in audit_notebook_parity() if check["status"] != "match"]
        self.assertEqual([], mismatches)


if __name__ == "__main__":
    unittest.main()
