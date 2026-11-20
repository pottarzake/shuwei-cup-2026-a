"""Retry an implicit step while preserving accepted substeps."""


class ConvergenceError(RuntimeError):
    """An implicit step did not meet its convergence criterion."""


def advance(model, dt_s, *, max_halvings=6, on_step=None, **kwargs):
    """Split a rejected step. Stop early when on_step returns False."""
    try:
        result = model.step(dt_s, **kwargs)
    except ConvergenceError:
        if max_halvings == 0:
            raise
        options = dict(max_halvings=max_halvings - 1, on_step=on_step, **kwargs)
        result = advance(model, dt_s / 2, **options)
        if result is None:
            return None
        return advance(model, dt_s / 2, **options)
    if on_step is not None and not on_step():
        return None
    return result
