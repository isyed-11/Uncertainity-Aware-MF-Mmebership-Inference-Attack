"""Presentation-only tests: no notebook cells, training, or model queries."""
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

from src.notebook_output import (
    compact_output, compact_dp_warnings, quiet_output, suppress_sklearn_parallel_warning,
    SKLEARN_PARALLEL_WARNING,
)


class OutputTests(unittest.TestCase):
    def test_dp_diagnostics_keep_first_occurrence_even_after_filter_reset(self):
        diagnostics = [
            ('Secure RNG turned off. This is perfectly fine for experimentation (test)',
             r'C:\project\.venv\Lib\site-packages\opacus\privacy_engine.py'),
            ('Optimal order is the largest alpha. (test)',
             r'C:\project\.venv\Lib\site-packages\opacus\accountants\analysis\rdp.py'),
            ('Full backward hook is firing when gradients are computed with respect '
             'to module outputs since no inputs require gradients. (test)',
             r'C:\project\src\dp_experiment.py'),
        ]
        visible = io.StringIO()
        with warnings.catch_warnings(), redirect_stderr(visible):
            original = warnings.showwarning
            with compact_dp_warnings():
                for _ in range(3):
                    warnings.resetwarnings()
                    warnings.simplefilter('always')
                    for message, filename in diagnostics:
                        warnings.warn_explicit(message, UserWarning, filename=filename, lineno=1)
                    warnings.warn('unrelated warning', UserWarning)
                # Matching words from a different source must remain visible.
                warnings.warn_explicit(diagnostics[1][0], UserWarning, filename='other.py', lineno=1)
            self.assertIs(warnings.showwarning, original)
        output = visible.getvalue()
        self.assertEqual(output.count(diagnostics[0][0]), 1)
        self.assertEqual(output.count(diagnostics[1][0]), 2)
        self.assertEqual(output.count(diagnostics[2][0]), 1)
        self.assertEqual(output.count('UserWarning: unrelated warning'), 3)

    def test_dp_warning_scope_restores_display_after_failure_and_resets_next_run(self):
        visible = io.StringIO()
        with warnings.catch_warnings(), redirect_stderr(visible):
            warnings.simplefilter('always')
            original = warnings.showwarning
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, 'visible failure'):
                    with compact_dp_warnings():
                        warnings.warn_explicit('Optimal order is the largest alpha.', UserWarning,
                            filename='/project/.venv/opacus/accountants/analysis/rdp.py', lineno=1)
                        raise RuntimeError('visible failure')
                self.assertIs(warnings.showwarning, original)
        self.assertEqual(visible.getvalue().count('Optimal order is the largest alpha.'), 2)

    def test_targeted_warning_stays_hidden_after_worker_filter_reset(self):
        visible = io.StringIO()
        with warnings.catch_warnings(), redirect_stderr(visible):
            suppress_sklearn_parallel_warning()
            hook = warnings.showwarning
            suppress_sklearn_parallel_warning()
            self.assertIs(hook, warnings.showwarning)
            # Reproduce the filter reset inside sklearn's worker wrapper.
            warnings.resetwarnings()
            warnings.simplefilter('always')
            warnings.warn(SKLEARN_PARALLEL_WARNING, UserWarning)
            warnings.warn('different warning remains visible', UserWarning)
        self.assertNotIn(SKLEARN_PARALLEL_WARNING, visible.getvalue())
        self.assertIn('different warning remains visible', visible.getvalue())

    def test_quiet_mode_discards_stdout_and_preserves_errors_and_stderr(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            with self.assertRaisesRegex(RuntimeError, 'visible failure'):
                with quiet_output():
                    print('verbose batches')
                    sys.stderr.write('visible diagnostic\n')
                    raise RuntimeError('visible failure')
            self.assertIs(sys.stdout, out)
            self.assertIs(sys.stderr, err)
        self.assertEqual(out.getvalue(), '')
        self.assertEqual(err.getvalue(), 'visible diagnostic\n')

    def test_progress_logged_and_distinct_warnings_visible_once(self):
        warning = 'example.py:144: UserWarning: repeated configuration warning\n  source()\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'output.log'
            visible_out, visible_err = io.StringIO(), io.StringIO()
            with patch('src.notebook_output._SEEN_WARNINGS', set()):
                with redirect_stdout(visible_out), redirect_stderr(visible_err):
                    for _ in range(2):
                        with compact_output(path):
                            print('batch progress')
                            sys.stderr.write(warning)
                            sys.stderr.write('unexpected diagnostic\n')
            self.assertEqual(visible_out.getvalue(), '')
            self.assertEqual(visible_err.getvalue().count('UserWarning'), 1)
            self.assertEqual(visible_err.getvalue().count('unexpected diagnostic'), 2)
            saved = path.read_text(encoding='utf-8')
            self.assertEqual(saved.count('UserWarning'), 1)
            self.assertIn('Repeated identical warnings omitted: 1', saved)
            self.assertEqual(saved.count('batch progress'), 2)

    def test_errors_propagate_and_streams_are_restored(self):
        original_out, original_err = sys.stdout, sys.stderr
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'visible failure'):
                with compact_output(Path(directory) / 'output.log'):
                    raise RuntimeError('visible failure')
        self.assertIs(sys.stdout, original_out)
        self.assertIs(sys.stderr, original_err)


if __name__ == '__main__':
    unittest.main()
