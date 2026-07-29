"""Standalone visual recognition and assembly planning for the E-topic puzzle."""


def solve_frame(*args, **kwargs):
    """Lazy public entry point so core modules stay independently testable."""
    from .pipeline import solve_frame as _solve_frame

    return _solve_frame(*args, **kwargs)


__all__ = ["solve_frame"]
