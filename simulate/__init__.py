from __future__ import annotations

from typing import Any, Dict


def simulate_eeg(params: Dict[str, float], forward_model: str = "jr", **kwargs: Any):
    """
    Unified dispatcher.
    Uses underlying module defaults by passing only kwargs provided by caller.
    """
    fm = forward_model.lower().replace("-", "_").strip()

    if fm in {"jr", "jansen_rit", "jansenrit"}:
        from .simulator import simulate_eeg as _sim
        return _sim(params=params, **kwargs)

    if fm in {"cmc", "canonical_microcircuit"}:
        from .cmc import simulate_eeg as _sim
        return _sim(params=params, **kwargs)

    raise ValueError(f"Unknown forward_model='{forward_model}'. Use 'jr' or 'cmc'.")
