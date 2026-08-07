from __future__ import annotations

import multiprocessing
import os
import tempfile
import time
import unittest

from agent.session_owner import SessionOwnerLease


def _hold_owner(workspace: str, ready: object, release: object) -> None:
    lease = SessionOwnerLease.acquire(workspace, "workspace-session")
    if lease is None:
        ready.put({"acquired": False})
        return
    ready.put({"acquired": True, "pid": lease.info.pid})
    release.wait(10)
    lease.release()


def _crash_owner(workspace: str, ready: object) -> None:
    """Acquire the lease, publish it, then exit without Python cleanup."""

    lease = SessionOwnerLease.acquire(workspace, "workspace-session")
    if lease is None:
        ready.put({"acquired": False})
        os._exit(2)
    lease.set_web_endpoint(54321, "crashed-token")
    ready.put({"acquired": True, "pid": lease.info.pid})
    ready.close()
    ready.join_thread()
    os._exit(0)


class SessionOwnerTests(unittest.TestCase):
    def test_owner_publishes_endpoint_and_releases_for_takeover(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = SessionOwnerLease.acquire(directory, "workspace-session")
            self.assertIsNotNone(first)
            assert first is not None
            first.set_web_endpoint(54321, "handshake-token")
            visible = SessionOwnerLease.read_existing(directory, "workspace-session")
            self.assertIsNotNone(visible)
            assert visible is not None
            self.assertEqual(visible.web_port, 54321)
            self.assertEqual(visible.web_token, "handshake-token")
            self.assertEqual(visible.pid, first.info.pid)

            first.release()
            self.assertIsNone(SessionOwnerLease.read_existing(directory, "workspace-session"))
            second = SessionOwnerLease.acquire(directory, "workspace-session")
            self.assertIsNotNone(second)
            assert second is not None
            second.release()

    def test_second_process_cannot_compete_until_owner_exits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context("spawn")
            ready = context.Queue()
            release = context.Event()
            child = context.Process(target=_hold_owner, args=(directory, ready, release))
            child.start()
            try:
                deadline = time.monotonic() + 10
                message = None
                while time.monotonic() < deadline and message is None:
                    try:
                        message = ready.get(timeout=0.2)
                    except Exception:
                        continue
                self.assertEqual(message and message.get("acquired"), True)
                contender = SessionOwnerLease.acquire(directory, "workspace-session")
                self.assertIsNone(contender)
                release.set()
                child.join(timeout=10)
                self.assertEqual(child.exitcode, 0)
                takeover = SessionOwnerLease.acquire(directory, "workspace-session")
                self.assertIsNotNone(takeover)
                if takeover is not None:
                    takeover.release()
            finally:
                release.set()
                if child.is_alive():
                    child.terminate()
                child.join(timeout=10)

    def test_abrupt_owner_exit_releases_lock_and_stale_metadata_does_not_block_takeover(self) -> None:
        """A crashed launcher must not strand the durable workflow.

        The sidecar can outlive an ungraceful process exit, but it is only
        descriptive.  The OS lock is authoritative, so a fresh launcher must
        be able to take ownership immediately and replace the stale endpoint.
        """

        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context("spawn")
            ready = context.Queue()
            child = context.Process(target=_crash_owner, args=(directory, ready))
            child.start()
            try:
                message = ready.get(timeout=10)
                self.assertEqual(message.get("acquired"), True)
                child.join(timeout=10)
                self.assertFalse(child.is_alive())
                self.assertEqual(child.exitcode, 0)

                stale = SessionOwnerLease.read_existing(directory, "workspace-session")
                self.assertIsNotNone(stale)
                assert stale is not None
                self.assertEqual(stale.pid, message["pid"])

                takeover = SessionOwnerLease.acquire(directory, "workspace-session")
                self.assertIsNotNone(takeover)
                assert takeover is not None
                self.assertNotEqual(takeover.info.owner_token, stale.owner_token)
                takeover.set_web_endpoint(54322, "replacement-token")

                visible = SessionOwnerLease.read_existing(directory, "workspace-session")
                self.assertIsNotNone(visible)
                assert visible is not None
                self.assertEqual(visible.pid, takeover.info.pid)
                self.assertEqual(visible.web_port, 54322)
                self.assertEqual(visible.web_token, "replacement-token")
                takeover.release()
            finally:
                if child.is_alive():
                    child.terminate()
                child.join(timeout=10)


if __name__ == "__main__":
    unittest.main()
