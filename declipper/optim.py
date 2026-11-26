from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple


def clip_scalar(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def project_bounds(x: Sequence[float], lo: Sequence[float], hi: Sequence[float]) -> List[float]:
    return [clip_scalar(float(v), float(lo[i]), float(hi[i])) for i, v in enumerate(x)]


def box_qp(
    n: int,
    energy_grad: Callable[[Sequence[float]], Tuple[float, List[float]]],
    lower: Sequence[float],
    upper: Sequence[float],
    x0: Sequence[float],
    *,
    lipschitz: float = 1.0,
    max_iter: int = 3000,
    grad_tol: float = 2e-7,
    trace_every: int = 50,
) -> Dict[str, object]:
    """Small spectral projected-gradient solver for box-constrained quadratic programs."""
    lo = [float(v) for v in lower]
    hi = [float(v) for v in upper]
    x = project_bounds(x0, lo, hi)
    energy, grad = energy_grad(x)
    initial_grad = max(1e-12, max(abs(g) for g in grad))
    alpha = 1.0 / max(1e-9, lipschitz)
    trace: List[Dict[str, float | int | str]] = []
    stop_reason = "max_iterations"
    converged = False
    pg_inf = math.inf
    violation = 0.0
    last_iteration = 0

    for iteration in range(1, max_iter + 1):
        last_iteration = iteration
        pg_inf = 0.0
        violation = 0.0
        for i in range(n):
            projected = clip_scalar(x[i] - grad[i], lo[i], hi[i])
            pg_inf = max(pg_inf, abs(x[i] - projected))
            violation = max(violation, lo[i] - x[i], x[i] - hi[i], 0.0)

        if iteration == 1 or iteration % trace_every == 0:
            trace.append(
                {
                    "iteration": iteration,
                    "energy": energy,
                    "projected_gradient_inf": pg_inf,
                    "constraint_violation": violation,
                    "step": alpha,
                }
            )

        # Scale the fixed-step KKT test by the initial objective gradient.
        if pg_inf <= grad_tol and violation <= 1e-8:
            stop_reason = "projected_gradient_tolerance"
            converged = True
            break
        if pg_inf <= grad_tol * max(1.0, initial_grad) and violation <= 1e-8:
            stop_reason = "relative_projected_gradient_tolerance"
            converged = True
            break

        accepted = False
        f0 = energy
        gdotd = 0.0
        x_new = x
        grad_new = grad
        energy_new = energy
        local_alpha = min(max(alpha, 1e-10), 1e8)
        for _ls in range(60):
            x_candidate = project_bounds(
                [x[i] - local_alpha * grad[i] for i in range(n)], lo, hi
            )
            d = [x_candidate[i] - x[i] for i in range(n)]
            gdotd = sum(grad[i] * d[i] for i in range(n))
            energy_candidate, grad_candidate = energy_grad(x_candidate)
            if energy_candidate <= f0 + 1e-4 * gdotd + 1e-11:
                x_new = x_candidate
                grad_new = grad_candidate
                energy_new = energy_candidate
                accepted = True
                break
            local_alpha *= 0.5
            if local_alpha < 1e-14:
                break

        if not accepted:
            stop_reason = "line_search_failed"
            break

        s = [x_new[i] - x[i] for i in range(n)]
        y = [grad_new[i] - grad[i] for i in range(n)]
        snorm = sum(v * v for v in s)
        sy = sum(s[i] * y[i] for i in range(n))
        step_inf = max((abs(v) for v in s), default=0.0)

        x, grad, energy = x_new, grad_new, energy_new
        if sy > 1e-14 * max(1.0, snorm):
            alpha = snorm / sy
        else:
            alpha = min(local_alpha * 1.5, 1e6)
        alpha = min(max(alpha, 1e-9), 1e6)

        if step_inf < 1e-13:
            pg_now = 0.0
            for i in range(n):
                projected = clip_scalar(x[i] - grad[i], lo[i], hi[i])
                pg_now = max(pg_now, abs(x[i] - projected))
            if pg_now < 1e-5 * max(1.0, initial_grad):
                stop_reason = "step_stalled_at_kkt_point"
                converged = True
                break

    final_violation = 0.0
    final_pg = 0.0
    for i in range(n):
        final_violation = max(final_violation, lo[i] - x[i], x[i] - hi[i], 0.0)
        final_pg = max(final_pg, abs(x[i] - clip_scalar(x[i] - grad[i], lo[i], hi[i])))
    trace.append(
        {
            "iteration": last_iteration,
            "energy": energy,
            "projected_gradient_inf": final_pg,
            "constraint_violation": final_violation,
            "stop_reason": stop_reason,  # type: ignore[dict-item]
        }
    )
    return {
        "x": x,
        "energy": energy,
        "iterations": last_iteration,
        "converged": converged,
        "stop_reason": stop_reason,
        "projected_gradient_inf": final_pg,
        "constraint_violation_max": final_violation,
        "trace": trace[-20:],
    }
