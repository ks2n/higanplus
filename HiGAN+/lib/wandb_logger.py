"""Optional Weights & Biases logger.

The repo has been TensorBoard-only forever, but for the recognizer-crop
experiment we want a single dashboard that compares baseline vs. each crop
mode (left_half / left_3q / char_aligned) across runs.  ``wandb`` is the
right tool for that: run grouping, config tracking, and image samples for
free.

We do not want a hard dependency, though.  Local smoke tests, CI, or anyone
running offline must keep working when ``wandb`` is not installed or no API
key is available.  This module exposes a tiny ``Logger`` facade that:

* silently no-ops when disabled, missing, or initialization fails;
* mirrors the same scalar keys the trainer already pushes to TensorBoard;
* accepts ``numpy``/``torch``/``matplotlib``-friendly inputs for images.

Configuration lives under the top-level ``wandb`` block in the YAML, e.g.::

    wandb:
      enabled: true
      project: higanplus
      entity: null         # default account
      tags: [recog-crop, char_aligned]
      group: feat-recog-random-crop
      mode: online         # online | offline | disabled

Anything missing falls back to safe defaults.  ``WANDB_API_KEY`` is read by
``wandb`` itself; we do not handle credentials here.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

try:  # ``wandb`` is optional.
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except Exception:  # pragma: no cover - optional dep
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False


def _opt_to_dict(opt) -> dict:
    """Best-effort conversion of a Munch/dict-like object to plain dict."""

    if hasattr(opt, "toDict"):
        try:
            return opt.toDict()
        except Exception:
            pass
    if isinstance(opt, dict):
        return dict(opt)
    out = {}
    for key in dir(opt):
        if key.startswith("_"):
            continue
        try:
            val = getattr(opt, key)
        except Exception:
            continue
        if callable(val):
            continue
        out[key] = val
    return out


class Logger:
    """No-op safe wrapper around the ``wandb`` Run object.

    All public methods accept the same arguments whether or not ``wandb`` is
    enabled, so call sites stay clean (no ``if wandb is not None`` blocks).
    """

    def __init__(self) -> None:
        self.enabled = False
        self._run = None

    def init(
        self,
        cfg_section,
        full_cfg=None,
        log_dir: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> "Logger":
        """Initialise wandb if enabled in config.

        Returns ``self`` for fluent use; failures degrade silently to the
        no-op state instead of crashing training.
        """

        if cfg_section is None:
            return self
        enabled = bool(getattr(cfg_section, "enabled", False))
        if not enabled:
            return self
        if not _WANDB_AVAILABLE:
            print("[wandb] python package not installed; skipping wandb logging.")
            return self

        project = getattr(cfg_section, "project", "higanplus")
        entity = getattr(cfg_section, "entity", None)
        group = getattr(cfg_section, "group", None)
        tags = getattr(cfg_section, "tags", None)
        mode = getattr(cfg_section, "mode", None) or os.environ.get("WANDB_MODE")

        cfg_dump = _opt_to_dict(full_cfg) if full_cfg is not None else None

        try:
            self._run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                group=group,
                tags=list(tags) if tags else None,
                config=cfg_dump,
                dir=log_dir,
                mode=mode,
                reinit=True,
            )
            self.enabled = True
            print(f"[wandb] logging to project={project!r} run={self._run.name!r}")
        except Exception as exc:  # pragma: no cover - network / auth issues
            print(f"[wandb] init failed ({exc}); continuing without wandb.")
            self.enabled = False
            self._run = None
        return self

    def log(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        """Push a scalar dict; silently dropped when disabled."""

        if not self.enabled or not metrics:
            return
        try:
            self._run.log(dict(metrics), step=step)
        except Exception:  # pragma: no cover
            pass

    def log_image(self, key: str, image: Any, step: Optional[int] = None,
                  caption: Optional[str] = None) -> None:
        if not self.enabled or image is None:
            return
        try:
            payload = {key: wandb.Image(image, caption=caption)}  # type: ignore[union-attr]
            self._run.log(payload, step=step)
        except Exception:  # pragma: no cover
            pass

    def finish(self) -> None:
        if not self.enabled:
            return
        try:
            self._run.finish()
        except Exception:  # pragma: no cover
            pass
        self.enabled = False
        self._run = None
