"""Dependency-gate checks that require no SDK, network or private data."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import ModuleType
from unittest.mock import patch

from experiments import sdk_environment as environment


class SDKEnvironmentTests(unittest.TestCase):
    def matching_fixture(self, directory):
        kit = Path(directory)/'kit'
        (kit/'rfq/rfq').mkdir(parents=True)
        engine = kit/'rfq/rfq/engine.py'
        engine.write_text('# Local dependency-gate fixture, not an SDK.\n')
        expected = Path(directory)/'expected.json'
        expected.write_text(json.dumps({'sha256': {'rfq/rfq/engine.py': hashlib.sha256(engine.read_bytes()).hexdigest()}}))
        return kit, expected

    def test_missing_sdk_is_an_explicit_dependency_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(environment, 'KIT', Path(directory)):
                self.assertFalse(environment.sdk_available())
                with self.assertRaisesRegex(ImportError, 'RFQ_SDK_ROOT'):
                    environment.require_sdk()

    def test_different_sdk_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            kit = Path(directory)
            (kit/'rfq/rfq').mkdir(parents=True)
            (kit/'rfq/rfq/engine.py').write_text('# A different exchange\n')
            with patch.object(environment, 'KIT', kit):
                self.assertTrue(environment.sdk_available())
                with self.assertRaisesRegex(ImportError, 'differs from the audited contract'):
                    environment.require_sdk()

    def test_external_sdk_source_keys_remain_relative_and_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            kit = Path(directory)
            with patch.object(environment, 'KIT', kit):
                path = kit/'rfq/rfq/engine.py'
                key = environment.source_key(path)
                self.assertEqual(key, 'inputs/official/kit/IMC/rfq/rfq/engine.py')
                self.assertFalse(Path(key).is_absolute())
                self.assertEqual(environment.source_path(key), path)

    def test_project_source_keys_round_trip(self):
        path = environment.ROOT/'submission/sdk_dealer.py'
        self.assertEqual(environment.source_key(path), 'submission/sdk_dealer.py')
        self.assertEqual(environment.source_path('submission/sdk_dealer.py'), path)

    def test_preflight_reports_unavailable_without_importing_another_sdk(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            variables = dict(os.environ, RFQ_SDK_ROOT=directory)
            result = subprocess.run([sys.executable, '-m', 'experiments.sdk_environment'],
                env=variables, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(json.loads(result.stdout)['available'])
            self.assertEqual(result.stderr, '')

    def test_preloaded_different_engine_is_rejected_despite_matching_disk_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            kit, expected = self.matching_fixture(directory)
            wrong = ModuleType('rfq.engine')
            wrong.__file__ = str(Path(directory)/'different-sdk/rfq/engine.py')
            with patch.object(environment, 'KIT', kit), patch.object(environment, 'EXPECTED', expected), patch.dict(sys.modules):
                for name in list(sys.modules):
                    if name == 'rfq' or name.startswith('rfq.'):
                        del sys.modules[name]
                sys.modules['rfq.engine'] = wrong
                with self.assertRaisesRegex(ImportError, 'Already imported rfq.engine'):
                    environment.require_sdk()
                self.assertIs(sys.modules['rfq.engine'], wrong)

    def test_namespace_with_any_foreign_search_origin_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            kit, expected = self.matching_fixture(directory)
            namespace = ModuleType('rfq')
            namespace.__path__ = [str(kit/'rfq/rfq'), str(Path(directory)/'different-sdk/rfq')]
            with patch.object(environment, 'KIT', kit), patch.object(environment, 'EXPECTED', expected), patch.dict(sys.modules):
                for name in list(sys.modules):
                    if name == 'rfq' or name.startswith('rfq.'):
                        del sys.modules[name]
                sys.modules['rfq'] = namespace
                with self.assertRaisesRegex(ImportError, 'Already imported rfq'):
                    environment.require_sdk()

    def test_correct_cached_modules_remain_and_configured_import_root_moves_first(self):
        with tempfile.TemporaryDirectory() as directory:
            kit, expected = self.matching_fixture(directory)
            package = ModuleType('rfq')
            package.__file__ = str(kit/'rfq/rfq/__init__.py')
            package.__path__ = [str(kit/'rfq/rfq')]
            engine = ModuleType('rfq.engine')
            engine.__file__ = str(kit/'rfq/rfq/engine.py')
            entries = [str(Path(directory)/'different-sdk'), str(kit/'rfq'), str(kit/'rfq')+'/.']
            with patch.object(environment, 'KIT', kit), patch.object(environment, 'EXPECTED', expected), patch.object(sys, 'path', entries), patch.dict(sys.modules):
                for name in list(sys.modules):
                    if name == 'rfq' or name.startswith('rfq.'):
                        del sys.modules[name]
                sys.modules.update({'rfq': package, 'rfq.engine': engine})
                self.assertEqual(environment.require_sdk(), kit)
                self.assertEqual(sys.path[0], str((kit/'rfq').resolve()))
                self.assertEqual(sum(Path(path).resolve() == (kit/'rfq').resolve() for path in sys.path), 1)
                self.assertIs(sys.modules['rfq.engine'], engine)
