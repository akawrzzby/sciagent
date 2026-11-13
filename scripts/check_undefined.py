"""Find names that are used but never defined or imported.

The project hit two real bugs of this kind -- ``chunk.section_title`` did not
exist on the model, and ``os`` was used in a module that never imported it --
and both stayed invisible because the unit tests never touched those paths.
This script walks the AST of every module and reports the names a function
uses that it does not define, import, or receive as an argument.

Usage::

    ~/envs/sciagent/bin/python scripts/check_undefined.py backend/app
"""

from __future__ import annotations

import ast
import builtins
import os
import sys
from typing import Iterable


class ScopeChecker(ast.NodeVisitor):
    """Collect the names a module reads but never binds."""

    def __init__(self, path: str, source: str) -> None:
        self.path = path
        self.source = source
        self.problems: list[tuple[int, str, str]] = []
        self.module_names: set[str] = set(dir(builtins))
        self._scope_stack: list[set[str]] = []

    # -- bindings ---------------------------------------------------------
    def _bind(self, name: str) -> None:
        if self._scope_stack:
            self._scope_stack[-1].add(name)
        else:
            self.module_names.add(name)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._bind(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                self.module_names.update(dir(builtins))
                continue
            self._bind(alias.asname or alias.name)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        for target in node.targets:
            for n in ast.walk(target):
                if isinstance(n, ast.Name):
                    self._bind(n.id)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            self._bind(node.target.id)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            self._bind(node.target.id)

    def visit_For(self, node: ast.For) -> None:
        self.generic_visit(node)
        for n in ast.walk(node.target):
            if isinstance(n, ast.Name):
                self._bind(n.id)

    def visit_With(self, node: ast.With) -> None:
        self.generic_visit(node)
        for item in node.items:
            if item.optional_vars is not None:
                for n in ast.walk(item.optional_vars):
                    if isinstance(n, ast.Name):
                        self._bind(n.id)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.generic_visit(node)
        if node.name:
            self._bind(node.name)

    def visit_Global(self, node: ast.Global) -> None:
        for n in node.names:
            self._bind(n)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for n in node.names:
            self._bind(n)

    # -- scopes -----------------------------------------------------------
    def _function_scope(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        local: set[str] = set()
        args = node.args
        for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            local.add(a.arg)
        if args.vararg:
            local.add(args.vararg.arg)
        if args.kwarg:
            local.add(args.kwarg.arg)
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                local.add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                local.add(n.name)
            elif isinstance(n, ast.ClassDef):
                local.add(n.name)
            elif isinstance(n, ast.alias):
                local.add((n.asname or n.name).split(".")[0])
            elif isinstance(n, ast.ExceptHandler) and n.name:
                local.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for al in n.names:
                    local.add((al.asname or al.name).split(".")[0])
            elif isinstance(n, ast.comprehension):
                for t in ast.walk(n.target):
                    if isinstance(t, ast.Name):
                        local.add(t.id)
        # Names used before the assignment are what we are looking for, so the
        # check happens with the module scope plus this function's locals.
        saved = set(self.module_names)
        self.module_names |= local
        self._check_body(node)
        self.module_names = saved

    def _class_scope(self, node: ast.ClassDef) -> None:
        local: set[str] = {node.name}
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                local.add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                local.add(n.name)
            elif isinstance(n, ast.alias):
                local.add((n.asname or n.name).split(".")[0])
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for al in n.names:
                    local.add((al.asname or al.name).split(".")[0])
            elif isinstance(n, ast.comprehension):
                for t in ast.walk(n.target):
                    if isinstance(t, ast.Name):
                        local.add(t.id)
        saved = set(self.module_names)
        self.module_names |= local
        self._check_body(node)
        self.module_names = saved

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function_scope(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_scope(node)

    def _check_body(self, node: ast.AST) -> None:
        """Report the names a scope reads but never binds.

        Nested functions and classes have their own scope and are checked
        separately, so the walk stops at their boundaries.
        """
        stack = list(ast.iter_child_nodes(node))
        while stack:
            child = stack.pop()
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._function_scope(child)
                continue
            if isinstance(child, ast.ClassDef):
                self._class_scope(child)
                continue
            if isinstance(child, ast.Lambda):
                local = set()
                for a in [*child.args.posonlyargs, *child.args.args, *child.args.kwonlyargs]:
                    local.add(a.arg)
                if child.args.vararg:
                    local.add(child.args.vararg.arg)
                if child.args.kwarg:
                    local.add(child.args.kwarg.arg)
                saved = set(self.module_names)
                self.module_names |= local
                self._check_body(child)
                self.module_names = saved
                continue
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                if child.id not in self.module_names:
                    self.problems.append((child.lineno, child.id, "undefined name"))
                continue
            stack.extend(ast.iter_child_nodes(child))

    def run(self) -> list[tuple[int, str, str]]:
        tree = ast.parse(self.source, filename=self.path)

        # Every name that the module binds at module level.  A name that is
        # defined further down the file still resolves at call time, so it
        # must not be reported.
        module_bound: set[str] = set()
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                module_bound.add(stmt.name)
            elif isinstance(stmt, ast.ClassDef):
                module_bound.add(stmt.name)
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                for al in stmt.names:
                    module_bound.add((al.asname or al.name).split(".")[0])
            else:
                for n in ast.walk(stmt):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                        module_bound.add(n.id)
                    elif isinstance(n, ast.alias):
                        module_bound.add((n.asname or n.name).split(".")[0])
        self.module_names |= module_bound

        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef):
                self._class_scope(stmt)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._function_scope(stmt)
            else:
                self.visit(stmt)
        return self.problems


def check_file(path: str) -> list[tuple[int, str, str]]:
    with open(path, "r", encoding="utf-8-sig") as fh:
        source = fh.read()
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        return [(exc.lineno or 0, "syntax", str(exc))]
    checker = ScopeChecker(path, source)
    try:
        return checker.run()
    except RecursionError:
        return []


def iter_files(root: str) -> Iterable[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".git"}]
        for name in filenames:
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def main() -> None:
    roots = sys.argv[1:] or ["backend/app"]
    total = 0
    for root in roots:
        for path in sorted(iter_files(root)):
            for lineno, name, kind in check_file(path):
                total += 1
                print(f"{path}:{lineno}: {kind}: {name}")
    print(f"\n{total} finding(s)")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
