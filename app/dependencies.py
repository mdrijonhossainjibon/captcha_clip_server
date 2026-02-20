from app.models.clip_solver import CLIPSolver

_solver: CLIPSolver | None = None


def get_solver() -> CLIPSolver:
    global _solver
    if _solver is None:
        # This triggers model loading if not already done by startup event
        _solver = CLIPSolver()
    return _solver
