"""Concise output; logs retain progress, distinct warnings and repeat counts."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from html import escape
from pathlib import Path
import re
import sys
import warnings
from threading import Lock

_SEEN_WARNINGS = set()
_LOGGED_WARNINGS = {}
_OUTPUT_LOCK = Lock()
_WARNING = re.compile(r'\b\w*Warning:')
SKLEARN_PARALLEL_WARNING = (
    '`sklearn.utils.parallel.delayed` should be used with'
    ' `sklearn.utils.parallel.Parallel` to make it possible to'
    ' propagate the scikit-learn configuration of the current thread to'
    ' the joblib workers.'
)


def suppress_sklearn_parallel_warning():
    """Hide only the known sklearn/joblib warning, even if filters are reset.

    Worker warning contexts can reset Python's filters. A targeted display hook
    keeps this warning hidden while forwarding all other warnings unchanged.
    Calling this again does not stack display hooks or change model execution.
    """
    warnings.filterwarnings('ignore', message=re.escape(SKLEARN_PARALLEL_WARNING),
                            category=UserWarning)
    if getattr(warnings.showwarning, '_suppresses_sklearn_parallel', False):
        return
    previous = warnings.showwarning

    def showwarning(message, category, filename, lineno, file=None, line=None):
        if issubclass(category, UserWarning) and str(message) == SKLEARN_PARALLEL_WARNING:
            return
        return previous(message, category, filename, lineno, file=file, line=line)

    showwarning._suppresses_sklearn_parallel = True
    warnings.showwarning = showwarning


@contextmanager
def compact_dp_warnings():
    """Show each of three known DP research diagnostics once per run.

    Keep the first message, including RNG/accountant limitations. Deduplicate
    only matching messages from the expected source files; unrelated warnings
    and exceptions retain their usual behavior. No files or settings change.
    """
    known = (
        ('/opacus/privacy_engine.py',
         'Secure RNG turned off. This is perfectly fine for experimentation'),
        ('/opacus/accountants/analysis/rdp.py',
         'Optimal order is the largest alpha.'),
        ('/src/dp_experiment.py',
         'Full backward hook is firing when gradients are computed with respect '
         'to module outputs since no inputs require gradients.'),
    )
    seen = set()
    with warnings.catch_warnings():
        previous = warnings.showwarning

        def showwarning(message, category, filename, lineno, file=None, line=None):
            if issubclass(category, UserWarning):
                origin = str(filename).replace('\\', '/').lower()
                for suffix, prefix in known:
                    if origin.endswith(suffix) and str(message).startswith(prefix):
                        if suffix in seen:
                            return
                        seen.add(suffix)
                        break
            return previous(message, category, filename, lineno, file=file, line=line)

        warnings.showwarning = showwarning
        yield


class _DiscardOutput:
    """Discard verbose stdout without retaining it in memory or on disk."""

    def write(self, text):
        return len(text)

    def flush(self):
        pass


@contextmanager
def quiet_output():
    """Hide batch chatter without files; leave stderr and exceptions visible."""
    with redirect_stdout(_DiscardOutput()):
        yield


class _WarningStream:
    """Retain distinct warnings and counts; show other stderr without filtering."""

    def __init__(self, log, visible):
        self.log, self.visible = log, visible
        self.log_seen = _LOGGED_WARNINGS.setdefault(str(Path(log.name).resolve()), set())
        self.repeated = 0

    def write(self, text):
        with _OUTPUT_LOCK:
            if _WARNING.search(text):
                if text not in self.log_seen:
                    self.log_seen.add(text)
                    self.log.write(text)
                else:
                    self.repeated += 1
                if text not in _SEEN_WARNINGS:
                    _SEEN_WARNINGS.add(text)
                    self.visible.write(text)
            else:
                self.log.write(text)
                # Unexpected stderr, including error text, is always visible.
                self.visible.write(text)
        return len(text)

    def finish(self):
        with _OUTPUT_LOCK:
            if self.repeated:
                self.log.write(f'[Repeated identical warnings omitted: {self.repeated:,}]\n')
            self.log.flush()

    def flush(self):
        self.log.flush()
        self.visible.flush()


@contextmanager
def compact_output(path):
    """Log stdout, retain distinct warnings/counts, and propagate errors.

    This changes display only. It does not install warning filters, modify
    sklearn settings, or alter computation/checkpoint fingerprints.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8', buffering=1) as log:
        stderr = _WarningStream(log, sys.stderr)
        try:
            with redirect_stdout(log), redirect_stderr(stderr):
                yield
        finally:
            stderr.finish()


class CellProgress:
    """Update a single notebook display rather than append progress lines."""

    def __init__(self):
        self.handle = None

    def update(self, message):
        from IPython.display import HTML, display
        content = HTML(f'<pre>{escape(str(message))}</pre>')
        if self.handle is None:
            self.handle = display(content, display_id=True)
        else:
            self.handle.update(content)
