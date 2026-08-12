import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent.workspace_registry import list_recent_workspaces, registry_path


class WorkspaceRegistryTests(unittest.TestCase):
    def test_available_projects_are_limited_before_stale_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            state_home = Path(directory) / "state"
            available = Path(directory) / "project"
            available.mkdir()
            with mock.patch.dict(os.environ, {"GA3BAD_STATE_HOME": str(state_home)}):
                path = registry_path()
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "workspaces": [
                                {
                                    "path": str(Path(directory) / "missing"),
                                    "last_opened": 20,
                                },
                                {"path": str(available), "last_opened": 10},
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                values = list_recent_workspaces(limit=1)

        self.assertEqual(len(values), 1)
        self.assertEqual(Path(values[0]["path"]), available)
        self.assertTrue(values[0]["available"])


if __name__ == "__main__":
    unittest.main()
