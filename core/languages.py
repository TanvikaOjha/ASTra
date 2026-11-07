"""Per-language tree-sitter configuration and normalised extraction.

This module is the *only* place that touches the tree-sitter API or knows
grammar node names. The ingestor (chunking) and graph builder consume the
normalised records defined here (``Definition``, ``ImportRecord``,
``CallRecord``), so adding a language means adding one ``LanguageSpec`` entry
(grammar loader + three queries + a few flags) and nothing else.

Verified against: tree-sitter 0.26.0 (``Query`` + ``QueryCursor`` API),
tree-sitter-python/javascript/typescript/go as resolved in Nov 2026.

Capture-name contract used by the queries
-----------------------------------------
definitions : ``@class.def/@class.name``, ``@function.def/@function.name``,
              ``@method.def/@method.name`` (+ Go: ``@method.receiver_type``,
              ``@method.receiver_var``)
imports     : ``@import.stmt`` (always), ``@import.module`` (always), and one of
              ``@import.name[/@import.alias]``, ``@import.default``,
              ``@import.namespace``, ``@import.wildcard``, ``@import.alias``
calls       : ``@call.site``, ``@call.name``, optional ``@call.receiver``
Captures starting with ``_`` are predicate helpers and are ignored.

Known limitations (also to be repeated in the README)
-----------------------------------------------------
* Purely syntactic. No type information, so ``obj.method()`` is only
  resolvable when ``obj`` is ``self``/``this``/an import alias.
* Python: ``import a.b`` binds the *full dotted text* ``a.b`` as the local
  name; a bare ``a.x()`` call after ``import a.b`` is not linked to ``a``.
* JS/TS: CommonJS support is limited to ``const x = require('m')`` and
  ``const {a, b: c} = require('m')``. ``module.exports = ...``,
  ``exports.foo = function(){}``, dynamic ``import()``, and TS path aliases
  (``tsconfig.json`` ``paths``) are not understood; aliased imports look like
  external packages.
* Go: the local name for an un-aliased import is the *last path segment*,
  which is only a heuristic (real package names can differ). Methods are
  named ``<file module>.<ReceiverType>.<Method>``, so a method defined in a
  different file than its type has a ``node_id`` in its own file's module
  and is linked to the type through ``receiver_type`` by the graph builder.
* Anonymous functions and callbacks are not definitions; calls inside them
  are attributed to the nearest enclosing *named* definition.
* A module name derives from the file path only (``a/b.py`` -> ``a.b``), so
  ``foo.js`` and ``foo.ts`` in the same folder would collide.
"""

from __future__ import annotations

import logging
import posixpath
import re
import threading
from bisect import bisect_right
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import PurePosixPath
from typing import Callable, Literal, Optional

import tree_sitter as ts

from core.models import ChunkKind

logger = logging.getLogger(__name__)

QueryName = Literal["definitions", "imports", "calls"]
BindingKind = Literal["module", "symbol", "wildcard"]

# --------------------------------------------------------------------------- #
# Grammar loaders (lazy, so one missing grammar package cannot break the rest)
# --------------------------------------------------------------------------- #


def _load_python() -> object:
    import tree_sitter_python as mod

    return mod.language()


def _load_javascript() -> object:
    import tree_sitter_javascript as mod

    return mod.language()


def _load_typescript() -> object:
    import tree_sitter_typescript as mod

    return mod.language_typescript()


def _load_tsx() -> object:
    import tree_sitter_typescript as mod

    return mod.language_tsx()


def _load_go() -> object:
    import tree_sitter_go as mod

    return mod.language()


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #

_PY_DEFINITIONS = """
(class_definition name: (identifier) @class.name) @class.def
(function_definition name: (identifier) @function.name) @function.def
"""

_PY_IMPORTS = """
(import_statement
  name: [(dotted_name) @import.module
         (aliased_import name: (dotted_name) @import.module alias: (identifier) @import.alias)]) @import.stmt
(import_from_statement
  module_name: [(dotted_name) (relative_import)] @import.module
  name: [(dotted_name) @import.name
         (aliased_import name: (dotted_name) @import.name alias: (identifier) @import.alias)]) @import.stmt
(import_from_statement
  module_name: [(dotted_name) (relative_import)] @import.module
  (wildcard_import) @import.wildcard) @import.stmt
"""

_PY_CALLS = """
(call function: (identifier) @call.name) @call.site
(call function: (attribute object: (_) @call.receiver attribute: (identifier) @call.name)) @call.site
"""

_JS_DEFINITIONS = """
(class_declaration name: (identifier) @class.name) @class.def
(function_declaration name: (identifier) @function.name) @function.def
(generator_function_declaration name: (identifier) @function.name) @function.def
(method_definition name: (_) @method.name) @method.def
(field_definition property: (_) @method.name value: [(arrow_function) (function_expression)]) @method.def
(variable_declarator name: (identifier) @function.name value: [(arrow_function) (function_expression)]) @function.def
"""

# TS/TSX: class names are ``type_identifier``; class fields are ``public_field_definition``;
# interfaces and abstract classes exist. Interfaces are treated as "class" definitions.
_TS_DEFINITIONS = """
(class_declaration name: (type_identifier) @class.name) @class.def
(abstract_class_declaration name: (type_identifier) @class.name) @class.def
(interface_declaration name: (type_identifier) @class.name) @class.def
(function_declaration name: (identifier) @function.name) @function.def
(generator_function_declaration name: (identifier) @function.name) @function.def
(method_definition name: (_) @method.name) @method.def
(public_field_definition name: (_) @method.name value: [(arrow_function) (function_expression)]) @method.def
(variable_declarator name: (identifier) @function.name value: [(arrow_function) (function_expression)]) @function.def
"""

_JS_IMPORTS = """
(import_statement source: (string (string_fragment) @import.module)) @import.stmt
(import_statement (import_clause (identifier) @import.default) source: (string (string_fragment) @import.module)) @import.stmt
(import_statement (import_clause (namespace_import (identifier) @import.namespace)) source: (string (string_fragment) @import.module)) @import.stmt
(import_statement
  (import_clause (named_imports (import_specifier name: (_) @import.name alias: (_)? @import.alias)))
  source: (string (string_fragment) @import.module)) @import.stmt
(variable_declarator
  name: (identifier) @import.namespace
  value: (call_expression function: (identifier) @_fn
           arguments: (arguments (string (string_fragment) @import.module)))
  (#eq? @_fn "require")) @import.stmt
(variable_declarator
  name: (object_pattern [(shorthand_property_identifier_pattern) @import.name
                         (pair_pattern key: (_) @import.name value: (identifier) @import.alias)])
  value: (call_expression function: (identifier) @_fn
           arguments: (arguments (string (string_fragment) @import.module)))
  (#eq? @_fn "require")) @import.stmt
"""

_JS_CALLS = """
(call_expression function: (identifier) @call.name) @call.site
(call_expression function: (member_expression object: (_) @call.receiver property: (property_identifier) @call.name)) @call.site
(new_expression constructor: (identifier) @call.name) @call.site
"""

# Go: structs and interfaces are "class" definitions; ``type X int`` is ignored.
_GO_DEFINITIONS = """
(function_declaration name: (identifier) @function.name) @function.def
(method_declaration
  receiver: (parameter_list
    (parameter_declaration
      name: (identifier)? @method.receiver_var
      type: [(type_identifier) @method.receiver_type
             (pointer_type (type_identifier) @method.receiver_type)
             (generic_type (type_identifier) @method.receiver_type)]))
  name: (field_identifier) @method.name) @method.def
(type_spec name: (type_identifier) @class.name type: [(struct_type) (interface_type)]) @class.def
"""

_GO_IMPORTS = """
(import_spec
  name: [(package_identifier) (dot) (blank_identifier)]? @import.alias
  path: (interpreted_string_literal (interpreted_string_literal_content) @import.module)) @import.stmt
"""

_GO_CALLS = """
(call_expression function: (identifier) @call.name) @call.site
(call_expression function: (selector_expression operand: (_) @call.receiver field: (field_identifier) @call.name)) @call.site
"""


# --------------------------------------------------------------------------- #
# Language spec + registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LanguageSpec:
    """Everything language-specific the pipeline needs (declarative)."""

    name: str
    extensions: tuple[str, ...]
    load_grammar: Callable[[], object]
    definition_query: str
    import_query: str
    call_query: str
    #: Fence/language key for ``RecursiveCharacterTextSplitter.from_language``
    #: (the enum member *name*, resolved by the ingestor).
    splitter_language: str
    #: Nodes that wrap a definition and belong in its line range
    #: (decorators, ``export``, ``const`` declarations, Go ``type`` keyword).
    range_wrapper_types: frozenset[str]
    comment_types: frozenset[str]
    docstring_style: Literal["python", "leading_comment"]
    line_comment_prefix: str
    #: File stems that name their directory's module (``__init__``, ``index``).
    index_module_names: tuple[str, ...]
    #: Receiver texts that mean "the enclosing class" (``self``, ``this``).
    self_receivers: frozenset[str]
    import_resolution: Literal["python", "js", "go"]
    #: Extensions tried, in order, when resolving a relative import to a file.
    resolve_extensions: tuple[str, ...] = ()


LANGUAGES: dict[str, LanguageSpec] = {
    "python": LanguageSpec(
        name="python",
        extensions=(".py",),
        load_grammar=_load_python,
        definition_query=_PY_DEFINITIONS,
        import_query=_PY_IMPORTS,
        call_query=_PY_CALLS,
        splitter_language="PYTHON",
        range_wrapper_types=frozenset({"decorated_definition"}),
        comment_types=frozenset({"comment"}),
        docstring_style="python",
        line_comment_prefix="#",
        index_module_names=("__init__",),
        self_receivers=frozenset({"self", "cls"}),
        import_resolution="python",
    ),
    "javascript": LanguageSpec(
        name="javascript",
        extensions=(".js", ".jsx", ".mjs", ".cjs"),
        load_grammar=_load_javascript,
        definition_query=_JS_DEFINITIONS,
        import_query=_JS_IMPORTS,
        call_query=_JS_CALLS,
        splitter_language="JS",
        range_wrapper_types=frozenset({"export_statement", "lexical_declaration", "variable_declaration"}),
        comment_types=frozenset({"comment"}),
        docstring_style="leading_comment",
        line_comment_prefix="//",
        index_module_names=("index",),
        self_receivers=frozenset({"this"}),
        import_resolution="js",
        resolve_extensions=(".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"),
    ),
    "typescript": LanguageSpec(
        name="typescript",
        extensions=(".ts", ".mts", ".cts"),
        load_grammar=_load_typescript,
        definition_query=_TS_DEFINITIONS,
        import_query=_JS_IMPORTS,
        call_query=_JS_CALLS,
        splitter_language="TS",
        range_wrapper_types=frozenset({"export_statement", "lexical_declaration", "variable_declaration"}),
        comment_types=frozenset({"comment"}),
        docstring_style="leading_comment",
        line_comment_prefix="//",
        index_module_names=("index",),
        self_receivers=frozenset({"this"}),
        import_resolution="js",
        resolve_extensions=(".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"),
    ),
    "tsx": LanguageSpec(
        name="tsx",
        extensions=(".tsx",),
        load_grammar=_load_tsx,
        definition_query=_TS_DEFINITIONS,
        import_query=_JS_IMPORTS,
        call_query=_JS_CALLS,
        splitter_language="TS",
        range_wrapper_types=frozenset({"export_statement", "lexical_declaration", "variable_declaration"}),
        comment_types=frozenset({"comment"}),
        docstring_style="leading_comment",
        line_comment_prefix="//",
        index_module_names=("index",),
        self_receivers=frozenset({"this"}),
        import_resolution="js",
        resolve_extensions=(".tsx", ".ts", ".jsx", ".js", ".mjs", ".cjs"),
    ),
    "go": LanguageSpec(
        name="go",
        extensions=(".go",),
        load_grammar=_load_go,
        definition_query=_GO_DEFINITIONS,
        import_query=_GO_IMPORTS,
        call_query=_GO_CALLS,
        splitter_language="GO",
        range_wrapper_types=frozenset({"type_declaration"}),
        comment_types=frozenset({"comment"}),
        docstring_style="leading_comment",
        line_comment_prefix="//",
        index_module_names=(),
        self_receivers=frozenset(),  # Go receivers are arbitrary names (see Definition.receiver_var)
        import_resolution="go",
    ),
}

_EXTENSION_MAP: dict[str, LanguageSpec] = {
    ext: spec for spec in LANGUAGES.values() for ext in spec.extensions
}

#: Filename suffixes that are source-like but must not be indexed.
_SKIPPED_SUFFIXES: tuple[str, ...] = (".d.ts", ".min.js", ".min.mjs", ".min.cjs")


def supported_extensions() -> frozenset[str]:
    """All file extensions that map to a supported language."""
    return frozenset(_EXTENSION_MAP)


def get_spec_for_path(path: str) -> Optional[LanguageSpec]:
    """Return the language spec for ``path`` or ``None`` if unsupported/skipped.

    Type-declaration files (``.d.ts``) and minified bundles are skipped
    because they carry no useful implementation to chunk or graph.
    """
    lowered = path.lower()
    if lowered.endswith(_SKIPPED_SUFFIXES):
        return None
    return _EXTENSION_MAP.get(PurePosixPath(lowered).suffix)


def module_name_for(spec: LanguageSpec, rel_path: str) -> str:
    """Derive a dotted module name from a repo-relative POSIX path.

    ``pkg/util.py`` -> ``pkg.util``; ``pkg/__init__.py`` -> ``pkg``;
    ``src/index.ts`` -> ``src``.
    """
    parts = list(PurePosixPath(rel_path).with_suffix("").parts)
    if len(parts) > 1 and parts[-1] in spec.index_module_names:
        parts.pop()
    return ".".join(parts)


# tree-sitter plumbing (cached; parsers are per-thread)

_thread_local = threading.local()


@lru_cache(maxsize=None)
def _language(name: str) -> ts.Language:
    """Build (once) the tree-sitter ``Language`` for a spec name."""
    return ts.Language(LANGUAGES[name].load_grammar())


@lru_cache(maxsize=None)
def _query(name: str, which: QueryName) -> ts.Query:
    """Compile (once) one of a language's queries."""
    spec = LANGUAGES[name]
    source = {
        "definitions": spec.definition_query,
        "imports": spec.import_query,
        "calls": spec.call_query,
    }[which]
    return ts.Query(_language(name), source)


def _parser(name: str) -> ts.Parser:
    """Return this thread's parser for ``name`` (parsers are not thread-safe)."""
    parsers: dict[str, ts.Parser] | None = getattr(_thread_local, "parsers", None)
    if parsers is None:
        parsers = {}
        _thread_local.parsers = parsers
    if name not in parsers:
        parsers[name] = ts.Parser(_language(name))
    return parsers[name]


def parse_source(spec: LanguageSpec, source: bytes) -> ts.Tree:
    """Parse ``source``. tree-sitter is error tolerant: check ``root_node.has_error``."""
    return _parser(spec.name).parse(source)


def validate_queries() -> None:
    """Compile every query of every language; raises ``tree_sitter.QueryError``.

    Used by tests and ``/health`` to fail fast when a grammar upgrade renames nodes.
    """
    for name in LANGUAGES:
        for which in ("definitions", "imports", "calls"):
            _query(name, which)  # type: ignore[arg-type]


def _matches(spec: LanguageSpec, which: QueryName, root: ts.Node) -> list[tuple[int, dict[str, list[ts.Node]]]]:
    """Run a query and return ``(pattern_index, captures)`` matches."""
    return ts.QueryCursor(_query(spec.name, which)).matches(root)


def _text(source: bytes, node: ts.Node) -> str:
    """Decode a node's source text (errors replaced)."""
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _first(caps: dict[str, list[ts.Node]], key: str) -> Optional[ts.Node]:
    nodes = caps.get(key)
    return nodes[0] if nodes else None


# Definitions


@dataclass(eq=False)
class Definition:
    """A class/function/method found in a file (normalised across languages)."""

    kind: ChunkKind
    name: str
    #: Dotted name relative to the module, unique within the file
    #: (``Class.method``; duplicates get ``@<start_line>`` appended).
    qualified_name: str
    parent_qualified_name: Optional[str]
    node: ts.Node = field(repr=False)
    #: ``node`` plus wrappers (decorators / ``export`` / ``const``).
    outer: ts.Node = field(repr=False)
    #: 1-based inclusive range of ``outer``, extended upward over a leading
    #: doc-comment block (JS/TS/Go) so the comment travels with its definition.
    start_line: int = 0
    end_line: int = 0
    signature: str = ""
    docstring: str = ""
    parent: Optional["Definition"] = field(default=None, repr=False)
    #: Go only: receiver type name and receiver variable name (``s`` in ``(s *S)``).
    receiver_type: Optional[str] = None
    receiver_var: Optional[str] = None


def outer_node(node: ts.Node, spec: LanguageSpec) -> ts.Node:
    """Expand ``node`` through wrapper nodes that hold *only* this definition.

    A wrapper is used only if it contains exactly one child of the same type,
    so ``const a = () => {}, b = () => {}`` and grouped Go ``type ( ... )``
    declarations are not over-extended.
    """
    current = node
    while True:
        parent = current.parent
        if parent is None or parent.type not in spec.range_wrapper_types:
            return current
        same = sum(1 for child in parent.named_children if child.type == current.type)
        if same != 1:
            return current
        current = parent


def _signature(node: ts.Node, outer: ts.Node, source: bytes) -> str:
    """Whitespace-normalised text from the start of ``outer`` to the body start."""
    body = node.child_by_field_name("body")
    if body is None:
        value = node.child_by_field_name("value")  # arrow/function expression in a declarator
        if value is not None:
            body = value.child_by_field_name("body")
    if body is not None and body.start_byte > outer.start_byte:
        raw = source[outer.start_byte : body.start_byte]
    else:
        raw = source[outer.start_byte : outer.end_byte].split(b"\n", 1)[0]
    return " ".join(raw.decode("utf-8", errors="replace").split())


_PY_STRING_RE = re.compile(r"""^[rRuUbBfF]{0,2}(\"\"\"|'''|"|')(.*)\1$""", re.DOTALL)


def _python_docstring(node: ts.Node, source: bytes) -> str:
    """First statement of the body if it is a plain string literal."""
    body = node.child_by_field_name("body")
    if body is None or not body.named_children:
        return ""
    first = body.named_children[0]
    if first.type != "expression_statement" or not first.named_children:
        return ""
    literal = first.named_children[0]
    if literal.type not in ("string", "concatenated_string"):
        return ""
    raw = _text(source, literal).strip()
    match = _PY_STRING_RE.match(raw)
    return (match.group(2) if match else raw).strip()


_COMMENT_MARKER_RE = re.compile(r"^\s*(/\*+|\*+/|\*|//+)\s?|\s*\*+/\s*$")


def _leading_comment_docstring(outer: ts.Node, spec: LanguageSpec, source: bytes) -> tuple[str, Optional[int]]:
    """Contiguous full-line comments directly above ``outer`` (JSDoc, Go doc comments).

    Returns ``(cleaned_text, first_comment_row)``; the row is 0-based and
    ``None`` when there is no leading comment.
    """
    lines: list[str] = []
    first_row: Optional[int] = None
    expected_row = outer.start_point.row
    prev = outer.prev_sibling
    while prev is not None and prev.type in spec.comment_types and prev.end_point.row + 1 >= expected_row:
        line_start = source.rfind(b"\n", 0, prev.start_byte) + 1
        if source[line_start : prev.start_byte].strip():  # trailing comment of another statement
            break
        lines.append(_text(source, prev))
        first_row = prev.start_point.row
        expected_row = prev.start_point.row
        prev = prev.prev_sibling
    cleaned: list[str] = []
    for block in reversed(lines):
        for line in block.splitlines():
            cleaned.append(_COMMENT_MARKER_RE.sub("", line).rstrip())
    return "\n".join(cleaned).strip(), first_row


def extract_definitions(spec: LanguageSpec, root: ts.Node, source: bytes) -> list[Definition]:
    """Return all definitions in a parsed file, ordered by position.

    Nesting is derived from byte ranges: a definition's parent is the nearest
    enclosing definition. A ``function`` directly inside a class becomes a
    ``METHOD``. Go methods have no syntactic parent; they are named
    ``<ReceiverType>.<Method>`` and linked to a same-file type if one exists.
    """
    raws: list[tuple[str, ts.Node, ts.Node, dict[str, list[ts.Node]]]] = []
    seen_ids: set[int] = set()
    for _, caps in _matches(spec, "definitions", root):
        for kind in ("class", "function", "method"):
            def_node = _first(caps, f"{kind}.def")
            name_node = _first(caps, f"{kind}.name")
            if def_node is None or name_node is None or def_node.id in seen_ids:
                continue
            seen_ids.add(def_node.id)
            raws.append((kind, def_node, name_node, caps))
            break
    raws.sort(key=lambda r: (r[1].start_byte, -r[1].end_byte))

    defs: list[Definition] = []
    stack: list[Definition] = []
    used_names: set[str] = set()
    for kind, def_node, name_node, caps in raws:
        while stack and not (
            stack[-1].node.start_byte <= def_node.start_byte and def_node.end_byte <= stack[-1].node.end_byte
        ):
            stack.pop()
        parent = stack[-1] if stack else None
        name = _text(source, name_node)

        receiver_type_node = _first(caps, "method.receiver_type")
        receiver_var_node = _first(caps, "method.receiver_var")
        receiver_type = _text(source, receiver_type_node) if receiver_type_node else None
        receiver_var = _text(source, receiver_var_node) if receiver_var_node else None

        if kind == "class":
            chunk_kind = ChunkKind.CLASS
        elif kind == "method" or (parent is not None and parent.kind is ChunkKind.CLASS):
            chunk_kind = ChunkKind.METHOD
        else:
            chunk_kind = ChunkKind.FUNCTION

        if receiver_type and parent is None:
            qualified = f"{receiver_type}.{name}"
        elif parent is not None:
            qualified = f"{parent.qualified_name}.{name}"
        else:
            qualified = name
        if qualified in used_names:  # redefinition / property setters / conditional defs
            qualified = f"{qualified}@{def_node.start_point.row + 1}"
        used_names.add(qualified)

        outer = outer_node(def_node, spec)
        start_row = outer.start_point.row
        if spec.docstring_style == "python":
            docstring = _python_docstring(def_node, source)
        else:
            docstring, first_comment_row = _leading_comment_docstring(outer, spec, source)
            if first_comment_row is not None:
                start_row = min(start_row, first_comment_row)  # doc comment belongs to the definition

        definition = Definition(
            kind=chunk_kind,
            name=name,
            qualified_name=qualified,
            parent_qualified_name=parent.qualified_name if parent else None,
            node=def_node,
            outer=outer,
            start_line=start_row + 1,
            end_line=outer.end_point.row + 1,
            signature=_signature(def_node, outer, source),
            docstring=docstring,
            parent=parent,
            receiver_type=receiver_type,
            receiver_var=receiver_var,
        )
        defs.append(definition)
        stack.append(definition)

    # Go: attach methods to a same-file type when one exists (cross-file links are the graph builder's job).
    type_names = {d.name: d for d in defs if d.kind is ChunkKind.CLASS and d.parent is None}
    for d in defs:
        if d.receiver_type and d.parent is None and d.receiver_type in type_names:
            d.parent = type_names[d.receiver_type]
            d.parent_qualified_name = d.parent.qualified_name
    return defs


class DefinitionIndex:
    """Find the innermost definition containing a byte offset (``None`` = module level)."""

    def __init__(self, definitions: list[Definition]) -> None:
        self._defs = sorted(definitions, key=lambda d: (d.node.start_byte, -d.node.end_byte))
        self._starts = [d.node.start_byte for d in self._defs]

    def innermost(self, byte_offset: int) -> Optional[Definition]:
        """Return the deepest definition whose node covers ``byte_offset``."""
        idx = bisect_right(self._starts, byte_offset) - 1
        if idx < 0:
            return None
        candidate: Optional[Definition] = self._defs[idx]
        # Climb the *syntactic* nesting chain only; Go methods attached to a type
        # are not lexically inside it, so check containment at every step.
        while candidate is not None and not (
            candidate.node.start_byte <= byte_offset < candidate.node.end_byte
        ):
            lexical_parent = candidate.parent
            if lexical_parent is not None and lexical_parent.node.start_byte <= candidate.node.start_byte and (
                candidate.node.end_byte <= lexical_parent.node.end_byte
            ):
                candidate = lexical_parent
            else:
                # Not nested: look further back for an earlier definition that covers the offset.
                idx -= 1
                candidate = self._defs[idx] if idx >= 0 else None
        return candidate


# Imports


@dataclass(frozen=True)
class ImportBinding:
    """A name an import statement introduces into the file's scope.

    ``kind``:
      * ``module``   - ``local_name`` refers to the module itself.
      * ``symbol``   - ``local_name`` refers to ``imported_name`` inside the module.
        For Python ``from pkg import x``, ``x`` may be a submodule rather than a symbol.
      * ``wildcard`` - everything public (Python ``*``, Go ``.`` imports); names unknown.
    """

    local_name: str
    imported_name: Optional[str]
    kind: BindingKind


@dataclass
class ImportRecord:
    """One imported module and the names it binds."""

    module: str
    line: int
    is_relative: bool
    bindings: list[ImportBinding] = field(default_factory=list)


def _go_default_local_name(module: str) -> str:
    """Last path segment (heuristic; real package names can differ, e.g. ``.../v2``)."""
    segments = [s for s in module.split("/") if s]
    if len(segments) > 1 and re.fullmatch(r"v\d+", segments[-1]):
        return segments[-2]
    return segments[-1] if segments else module


def extract_imports(spec: LanguageSpec, root: ts.Node, source: bytes) -> list[ImportRecord]:
    """Return normalised imports, one record per (statement, module)."""
    records: dict[tuple[int, str], ImportRecord] = {}
    for _, caps in _matches(spec, "imports", root):
        stmt = _first(caps, "import.stmt")
        module_node = _first(caps, "import.module")
        if stmt is None or module_node is None:
            continue
        module = _text(source, module_node)
        key = (stmt.start_byte, module)
        record = records.get(key)
        if record is None:
            record = ImportRecord(
                module=module,
                line=stmt.start_point.row + 1,
                is_relative=module.startswith(".") and spec.import_resolution != "go",
            )
            records[key] = record

        name = _first(caps, "import.name")
        alias = _first(caps, "import.alias")
        name_text = _text(source, name) if name else None
        alias_text = _text(source, alias) if alias else None
        binding: Optional[ImportBinding] = None

        if _first(caps, "import.wildcard") is not None:
            binding = ImportBinding("*", None, "wildcard")
        elif name_text is not None:
            binding = ImportBinding(alias_text or name_text, name_text, "symbol")
        elif _first(caps, "import.default") is not None:
            binding = ImportBinding(_text(source, _first(caps, "import.default")), "default", "symbol")  # type: ignore[arg-type]
        elif _first(caps, "import.namespace") is not None:
            binding = ImportBinding(_text(source, _first(caps, "import.namespace")), None, "module")  # type: ignore[arg-type]
        elif spec.import_resolution == "python":
            binding = ImportBinding(alias_text or module, None, "module")
        elif spec.import_resolution == "go":
            if alias_text == ".":
                binding = ImportBinding("*", None, "wildcard")
            elif alias_text == "_":
                binding = None  # side-effect import, binds nothing
            else:
                binding = ImportBinding(alias_text or _go_default_local_name(module), None, "module")
        # JS bare ``import 'x'`` has no bindings; the record itself is still kept.

        if binding is not None and binding not in record.bindings:
            record.bindings.append(binding)
    return sorted(records.values(), key=lambda r: r.line)


def submodule_name(module: str, name: str) -> str:
    """Python: dotted module string for ``from <module> import <name>`` as a submodule."""
    return f"{module}{name}" if module.endswith(".") else f"{module}.{name}"


def import_candidates(
    spec: LanguageSpec,
    importer_path: str,
    module: str,
    *,
    go_module: Optional[str] = None,
) -> list[str]:
    """Candidate repo-relative targets for an import, in priority order.

    Returns *files* for Python/JS/TS and a *directory* for Go (a Go import
    names a package = a directory). The graph builder keeps only candidates
    that exist in the index; an empty list means "external or unresolvable".
    ``go_module`` is the ``module`` line of ``go.mod`` (see ``parse_go_mod``);
    without it Go imports are never resolved to local code.
    """
    if spec.import_resolution == "python":
        if module.startswith("."):
            level = len(module) - len(module.lstrip("."))
            rest = module.lstrip(".")
            base = PurePosixPath(importer_path).parent
            for _ in range(level - 1):
                base = base.parent
            if not rest:  # ``from . import x`` names the package directory itself
                return [str(base / "__init__.py")]
            target = base / rest.replace(".", "/")
        else:
            target = PurePosixPath(module.replace(".", "/"))
        return [f"{target}.py", f"{target}/__init__.py"]

    if spec.import_resolution == "js":
        if not module.startswith("."):
            return []  # bare specifier: external package (or an unsupported path alias)
        base = posixpath.normpath(posixpath.join(posixpath.dirname(importer_path), module))
        if base.startswith(".."):
            return []
        candidates: list[str] = []
        if PurePosixPath(base).suffix in set(spec.resolve_extensions):
            candidates.append(base)
        # ESM-style TypeScript imports write ``./x.js`` for a file that is ``x.ts``.
        stem, ext = posixpath.splitext(base)
        if ext in (".js", ".jsx", ".mjs", ".cjs"):
            candidates += [f"{stem}.ts", f"{stem}.tsx"]
        candidates += [f"{base}{e}" for e in spec.resolve_extensions]
        candidates += [f"{base}/index{e}" for e in spec.resolve_extensions]
        return list(dict.fromkeys(candidates))

    if spec.import_resolution == "go":
        if not go_module or not (module == go_module or module.startswith(go_module + "/")):
            return []
        return [module[len(go_module) :].lstrip("/") or "."]

    return []


_GO_MOD_RE = re.compile(r"^\s*module\s+(\S+)", re.MULTILINE)


def parse_go_mod(text: str) -> Optional[str]:
    """Return the module path from ``go.mod`` text, or ``None``."""
    match = _GO_MOD_RE.search(text)
    return match.group(1).strip('"') if match else None


# Calls

_SIMPLE_RECEIVER_RE = re.compile(r"^[A-Za-z_$][\w$]*(\.[A-Za-z_$][\w$]*)*$")


@dataclass(frozen=True)
class CallRecord:
    """A call site. Resolution happens later; this is purely syntactic."""

    name: str
    receiver: Optional[str]
    #: True when the receiver is a plain (dotted) identifier such as ``self`` or ``os.path``;
    #: False for expressions like ``foo().bar`` where resolution is not attempted.
    receiver_is_simple: bool
    line: int
    start_byte: int


def extract_calls(spec: LanguageSpec, root: ts.Node, source: bytes) -> list[CallRecord]:
    """Return every call site (``f()``, ``obj.f()``, ``new F()``) in a parsed file."""
    calls: list[CallRecord] = []
    seen: set[tuple[int, int]] = set()
    for _, caps in _matches(spec, "calls", root):
        site = _first(caps, "call.site")
        name = _first(caps, "call.name")
        if site is None or name is None or (site.start_byte, site.end_byte) in seen:
            continue
        seen.add((site.start_byte, site.end_byte))
        receiver_node = _first(caps, "call.receiver")
        receiver = _text(source, receiver_node) if receiver_node else None
        calls.append(
            CallRecord(
                name=_text(source, name),
                receiver=receiver,
                receiver_is_simple=bool(receiver and _SIMPLE_RECEIVER_RE.match(receiver)),
                line=site.start_point.row + 1,
                start_byte=site.start_byte,
            )
        )
    calls.sort(key=lambda c: c.start_byte)
    return calls


# Smoke demo: ``python -m core.languages``

_DEMO: dict[str, tuple[str, str]] = {
    "python": (
        "pkg/mod.py",
        '''import os
from .util import helper as h
@decorator
class A(B):
    """Class doc."""
    def run(self, x):
        """Run it."""
        self.step(x)
        return h(x) + os.getpid()
    def step(self, x): pass
def top(): A().run(1)
''',
    ),
    "typescript": (
        "src/app.ts",
        '''import { load as ld } from "./store";
/** Service doc. */
export class Svc { run(x: number): number { return this.go(ld(x)); } go = (y: number) => y; }
export const make = () => new Svc();
''',
    ),
    "go": (
        "cmd/main.go",
        '''package main
import ("fmt"; str "strings")
type S struct{}
// Do does it.
func (s *S) Do() { s.other(); fmt.Println(str.ToUpper("x")) }
func main() { (&S{}).Do() }
''',
    ),
}


def _demo() -> None:
    """Parse the embedded samples and log what was extracted."""
    from config import configure_logging

    configure_logging()
    validate_queries()
    logger.info("All queries compile for: %s", ", ".join(LANGUAGES))
    for lang, (path, code) in _DEMO.items():
        spec = get_spec_for_path(path)
        assert spec is not None and spec.name == lang
        data = code.encode()
        tree = parse_source(spec, data)
        logger.info("== %s (%s) module=%s has_error=%s", lang, path, module_name_for(spec, path), tree.root_node.has_error)
        defs = extract_definitions(spec, tree.root_node, data)
        index = DefinitionIndex(defs)
        for d in defs:
            logger.info("  def %-8s %-14s L%d-%d sig=%r doc=%r", d.kind.value, d.qualified_name, d.start_line, d.end_line, d.signature, d.docstring)
        for imp in extract_imports(spec, tree.root_node, data):
            logger.info("  import %r rel=%s -> %s", imp.module, imp.is_relative, [(b.local_name, b.imported_name, b.kind) for b in imp.bindings])
        for call in extract_calls(spec, tree.root_node, data):
            owner = index.innermost(call.start_byte)
            logger.info("  call %s.%s L%d in %s", call.receiver, call.name, call.line, owner.qualified_name if owner else "<module>")


if __name__ == "__main__":
    _demo()