"""Dependency-neutral diagnostics shared by model execution and the pipeline."""

import sys
import time

_DIAG_LAST = 0.0


def runtime_diagnostic(message):
    """Write one timestamped runtime diagnostic to stderr."""
    global _DIAG_LAST
    now = time.time()
    gap = f' +{now - _DIAG_LAST:.1f}s' if _DIAG_LAST else ''
    _DIAG_LAST = now
    timestamp = time.strftime('%H:%M:%S')
    print(f'[{timestamp}]{gap} {message}', file=sys.stderr, flush=True)
