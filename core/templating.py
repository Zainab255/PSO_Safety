"""Placeholder substitution for prompt templates taken from the config."""

from __future__ import annotations

import re

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def fill_template(template: str, **values: str) -> str:
    """Replace ``{name}`` placeholders in a single pass.

    Unlike ``str.format`` this leaves other braces (e.g. the literal JSON
    examples in the judge prompts) untouched and never re-expands text that
    came from a substituted value.
    """
    return _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), template)
