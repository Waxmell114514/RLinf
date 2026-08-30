"""Efference-copy probing for pretrained VLAs on LIBERO (inference only).

This package is self-contained: nothing under ``rlinf/`` is modified.  Hidden
states are captured with a forward hook, and hijacks are applied to the action
tensor between the policy and the environment.

Modules split by dependency so the analysis half runs without torch/LIBERO:

* :mod:`.hijack`, :mod:`.probes`, :mod:`.datasets`, :mod:`.figures` --
  numpy/pandas/scikit-learn only.
* :mod:`.capture`, :mod:`.harness` -- require torch and the RLinf runtime.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
