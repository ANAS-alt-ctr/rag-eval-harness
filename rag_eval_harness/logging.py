"""Step-level logging.

Every query the pipeline answers is appended as one JSON object per line
(JSONL) to the configured log file, and kept in memory on the run so the
API/UI can show the trace: the question, the retrieved chunks with their
scores, the generated answer, its citations, tokens, latency and cost.
This is the observability layer a recruiter or interviewer can inspect:
it shows *why* the system answered what it did, not just the answer.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field


@dataclass
class StepLog:
    """In-memory trace of pipeline runs, optionally mirrored to disk."""

    log_file: str | None = None
    steps: list[dict] = field(default_factory=list)

    def record(self, step: str, detail: str = "", **extra) -> None:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "step": step,
            "detail": detail,
            **extra,
        }
        self.steps.append(entry)
        if self.log_file:
            try:
                os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)
                with open(self.log_file, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry) + "\n")
            except OSError:
                # Logging must never break the pipeline.
                pass

    def as_dicts(self) -> list[dict]:
        return list(self.steps)
