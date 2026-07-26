from types import SimpleNamespace
import unittest

from agent.tui_commands import COMMAND_SPECS, matching_commands


class CommandAvailabilityTests(unittest.TestCase):
    def test_palette_omits_foreign_commands_and_contextually_hides_mutations(self):
        names = {item.name for item in COMMAND_SPECS}
        self.assertFalse({"/ide", "/vim", "/experimental", "/sandbox-add-read-dir"} & names)

        running = SimpleNamespace(status="running", running=True, undo_available=False)
        visible = {item.name for item in matching_commands("", limit=100, snapshot=running)}
        self.assertIn("/pause", visible)
        self.assertNotIn("/resume", visible)
        self.assertNotIn("/undo", visible)

        paused = SimpleNamespace(status="paused", running=False, undo_available=True)
        visible = {item.name for item in matching_commands("", limit=100, snapshot=paused)}
        self.assertIn("/resume", visible)
        self.assertIn("/undo", visible)
        self.assertNotIn("/pause", visible)


if __name__ == "__main__":
    unittest.main()
