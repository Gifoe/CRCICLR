import importlib.util
import unittest
from pathlib import Path


SOURCE = Path(__file__).with_name("audit_native_pt_eegnet_v4.py")
SPEC = importlib.util.spec_from_file_location("audit_native_pt_eegnet_v4", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NativePtV4PatchTests(unittest.TestCase):
    def test_pinned_patch_compiles_and_uses_distinct_split_domains(self):
        source = MODULE.transformed_source()
        self.assertIn('data["split_sha256"] != frozen_task_split', source)
        self.assertIn('search_data["split_sha256"] != record["split_sha256"]', source)
        self.assertIn('"checkpoint_native_pt_v3" / f"epoch_{epoch:03d}.V4.FAIL_CLOSED.json"', source)
        self.assertIn('"superseded_v3_fail_closed_sha256"', source)
        self.assertNotIn(
            'if frozen_recipe != record["recipe"] or data["split_sha256"] != record["split_sha256"]:',
            source,
        )
        compile(source, str(SOURCE), "exec")


if __name__ == "__main__":
    unittest.main()
