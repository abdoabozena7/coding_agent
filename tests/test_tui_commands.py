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
            [item.name for item in matches[:4]],
            ["/plan", "/model", "/settings", "/help"],
        )
        self.assertGreater(len(matches), 9)

    def test_legacy_mode_command_is_hidden_from_interactive_palette(self):
        running = SimpleNamespace(status="running", running=True, undo_available=False)
        mode = next(item for item in COMMAND_SPECS if item.name == "/mode")
        availability = command_availability(mode, running)
        self.assertFalse(availability.visible)
        self.assertFalse(availability.enabled)
        self.assertIn("automatically", availability.reason.casefold())

    def test_running_palette_exposes_explicit_sleep_actions(self):
        running = SimpleNamespace(status="running", running=True, undo_available=False)
        names = {item.name for item in matching_commands("/sleep", snapshot=running)}
        self.assertTrue({"/sleep on", "/sleep off", "/sleep status"} <= names)

    def test_recovery_palette_keeps_sleep_and_stop_discoverable(self):
        recovering = SimpleNamespace(status="recovering", running=True, undo_available=False)
        names = {item.name for item in matching_commands("", snapshot=recovering)}
        self.assertTrue({"/sleep on", "/sleep off", "/sleep status", "/stop"} <= names)


if __name__ == "__main__":
    unittest.main()
