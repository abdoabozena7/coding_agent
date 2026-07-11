"""Memory-bounded workspace traversal shared by listing and search tools."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ._security import (
    MAX_TRAVERSAL_DEPTH,
    MAX_TRAVERSAL_ENTRIES,
    ToolSecurityError,
    is_sensitive_path,
    resolve_workspace_path,
)


@dataclass(frozen=True)
class WalkEntry:
    path: Path
    resolved: Path
    is_directory: bool
    is_symlink: bool
    depth: int


class BoundedWalker:
    """Iterative DFS that never materialises an unbounded directory.

    ``os.walk`` builds complete ``dirs`` and ``files`` lists for each directory
    before callers can apply a cap.  A single hostile directory can therefore
    consume arbitrary memory.  This scanner stops while iterating ``scandir``
    and retains at most ``max_entries`` candidates across all pending work.
    """

    def __init__(
        self,
        base: Path,
        *,
        ignore: set[str],
        max_entries: int = MAX_TRAVERSAL_ENTRIES,
        max_depth: int = MAX_TRAVERSAL_DEPTH,
    ) -> None:
        self.base = base
        self.ignore = ignore
        self.max_entries = max(1, max_entries)
        self.max_depth = max(0, max_depth)
        self.visited = 0
        self.truncated = False

    def __iter__(self) -> Iterator[WalkEntry]:
        stack: list[tuple[Path, int]] = [(self.base, 0)]
        while stack:
            directory, depth = stack.pop()
            try:
                canonical = resolve_workspace_path(
                    str(directory), allow_workspace=True, must_exist=True
                )
            except ToolSecurityError:
                continue
            if canonical != directory or not canonical.is_dir() or is_sensitive_path(canonical):
                continue
            if depth >= self.max_depth:
                # We deliberately do not peek below the limit; report that the
                # traversal may be incomplete rather than implying otherwise.
                self.truncated = True
                continue

            children: list[WalkEntry] = []
            hit_cap = False
            try:
                with os.scandir(directory) as iterator:
                    for raw in iterator:
                        if self.visited >= self.max_entries:
                            self.truncated = True
                            hit_cap = True
                            break
                        self.visited += 1
                        if raw.name in self.ignore:
                            continue

                        candidate = directory / raw.name
                        try:
                            resolved = resolve_workspace_path(
                                str(candidate), allow_workspace=True, must_exist=True
                            )
                        except ToolSecurityError:
                            continue
                        if is_sensitive_path(resolved):
                            continue

                        try:
                            symlink = raw.is_symlink()
                            is_directory = resolved.is_dir()
                        except OSError:
                            continue
                        children.append(
                            WalkEntry(
                                path=candidate,
                                resolved=resolved,
                                is_directory=is_directory,
                                is_symlink=symlink,
                                depth=depth + 1,
                            )
                        )
            except OSError:
                continue

            directories: list[tuple[Path, int]] = []
            for child in sorted(children, key=lambda item: item.path.name):
                yield child
                if child.is_directory and not child.is_symlink:
                    directories.append((child.path, child.depth))
            stack.extend(reversed(directories))
            if hit_cap:
                return
