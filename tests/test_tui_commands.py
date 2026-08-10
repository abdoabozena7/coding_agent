from types import SimpleNamespace
import unittest

from agent.tui_commands import COMMAND_SPECS, command_availability, matching_commands


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

    def test_idle_palette_is_route_neutral_and_keeps_plan_discoverable(self):
        idle = SimpleNamespace(status="idle", running=False, undo_available=False)
        matches = matching_commands("/", snapshot=idle)
        self.assertEqual(
            [item.name for item in matches[:5]],
            ["/plan", "/live", "/show-diff", "/advanced-tracing", "/settings"],
        )
        self.assertEqual(
            {item.name for item in COMMAND_SPECS},
            {
                "/plan", "/live", "/show-diff", "/advanced-tracing", "/settings", "/pause",
                "/resume", "/stop", "/undo", "/help", "/quit",
            },
        )

    def test_removed_commands_do_not_exist_in_interactive_metadata(self):
        running = SimpleNamespace(status="running", running=True, undo_available=False)
        names = {item.name for item in matching_commands("", snapshot=running)}
        self.assertFalse({"/mode", "/model", "/sleep", "/trace", "/agents"} & names)

    def test_running_palette_keeps_only_direct_recovery_controls(self):
        running = SimpleNamespace(status="running", running=True, undo_available=False)
        names = {item.name for item in matching_commands("", snapshot=running)}
        self.assertTrue({"/pause", "/stop", "/live", "/show-diff", "/advanced-tracing", "/settings"} <= names)
        self.assertNotIn("/resume", names)

    def test_recovery_palette_keeps_resume_and_stop_discoverable(self):
        recovering = SimpleNamespace(status="recovering", running=True, undo_available=False)
        names = {item.name for item in matching_commands("", snapshot=recovering)}
        self.assertTrue({"/pause", "/stop", "/live", "/show-diff", "/advanced-tracing"} <= names)


if __name__ == "__main__":
    unittest.main()
