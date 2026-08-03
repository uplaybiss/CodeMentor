"""Prompt templates loaded from YAML files in this package.

Each ``<agent>.yaml`` file carries a ``system`` key — the system prompt used by
the corresponding agent. Templates use ``$placeholder`` (``string.Template``)
syntax so JSON examples inside the prompt never collide with ``format`` braces.
"""

from prompts.loader import get_prompt, render_prompt

__all__ = ["get_prompt", "render_prompt"]
