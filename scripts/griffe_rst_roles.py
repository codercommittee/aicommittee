"""A Griffe extension for the documentation build: Sphinx roles in docstrings become cross-references.

The docstrings write ``:class:`~codercommittee.protocols.Model``` and ``:func:`compare```, which
``help()`` and pydoc show as written. mkdocstrings renders Markdown, so this extension rewrites
every role into the autorefs form ``[Model][codercommittee.protocols.Model]`` before rendering;
a bare name such as ``:class:`Task``` becomes ``[Task][]`` and resolves from the object's scope.
"""

from __future__ import annotations

import re
from typing import Any

from griffe import Extension, Module, Object

ROLE = re.compile(r":(?:class|func|mod|meth|attr|data|exc|obj):`(~?)([A-Za-z_][\w.]*)`")
LITERAL_BLOCK = re.compile(r"::$", re.MULTILINE)


def rewrite(text: str) -> str:
    """Every Sphinx role in ``text`` as an autorefs cross-reference; a literal block's ``::`` as a colon.

    The indented block that follows ``::`` is already a code block in Markdown; only the second
    colon would show.
    """
    text = LITERAL_BLOCK.sub(":", text)

    def replace(match: re.Match[str]) -> str:
        tilde, path = match.groups()
        label = path.rpartition(".")[2] if tilde else path
        if "." in path:
            return f"[{label}][{path}]"
        return f"[{label}][]"

    return ROLE.sub(replace, text)


class RstRoles(Extension):
    """Rewrite the Sphinx roles of every docstring once the package is loaded."""

    def on_package(self, *, pkg: Module, **kwargs: Any) -> None:
        """Walk the package and rewrite each docstring."""
        self._walk(pkg)

    def _walk(self, obj: Object) -> None:
        if obj.docstring is not None and ":" in obj.docstring.value:  # a role or a literal block
            obj.docstring.value = rewrite(obj.docstring.value)
        for member in obj.members.values():
            if isinstance(member, Object):  # aliases point at objects documented elsewhere
                self._walk(member)
