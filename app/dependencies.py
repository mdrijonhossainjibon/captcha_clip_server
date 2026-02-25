from app.models.clip_solver import MobileCLIPSolver

_solver: MobileCLIPSolver | None = None


def get_solver() -> MobileCLIPSolver:
    global _solver
    if _solver is None:
        _solver = MobileCLIPSolver()
    return _solver
