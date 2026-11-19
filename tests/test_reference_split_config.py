"""Reference-file regression checks; no notebook cells or experiments run."""
from contextlib import chdir
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.expanded_data import reference_split_config


EXPECTED = dict(validation_size=10000, test_size=10000,
    mia_nonmember_size=10000, mia_member_size=10000, seed=42)


class ReferenceSplitConfigTests(unittest.TestCase):
    def test_default_does_not_depend_on_notebook_name_or_working_directory(self):
        with TemporaryDirectory() as directory, chdir(directory):
            self.assertEqual(reference_split_config(), EXPECTED)

    def test_explicit_renamed_notebook_is_parsed_without_execution(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'renamed_notebook.ipynb'
            path.write_text(json.dumps(dict(cells=[dict(id='overfit-imports', source=[
                "raise RuntimeError('This notebook must not execute')\n",
                'OVERFIT_SEED = 42\n',
                'SPLIT_CONFIG = dict(validation_size=10000, test_size=10000,\n',
                '    mia_nonmember_size=10000, mia_member_size=10000, seed=OVERFIT_SEED)\n',
            ])])), encoding='utf-8')
            self.assertEqual(reference_split_config(path), EXPECTED)

    def test_explicit_missing_file_is_not_silently_replaced(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                reference_split_config(Path(directory) / 'missing.ipynb')

    def test_json_schema_rejects_missing_or_invalid_settings(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'splits.json'
            for value in ({}, dict(EXPECTED, seed=True), dict(EXPECTED, test_size=-1)):
                with self.subTest(value=value):
                    path.write_text(json.dumps(value), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        reference_split_config(path)


if __name__ == '__main__':
    unittest.main()
