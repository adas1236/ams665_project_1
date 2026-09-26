import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from typing import List

from src.util.chebyshev_utils import chebyshev_matrix, chebyshev_coefficients, clenshaw, solve_zero_boundary
from src.util.inhomogeneous_utils import boundary_lift


def exact_solution(problem: int, x: np.ndarray) -> np.ndarray:
    if problem == 5:
        return np.cos(np.pi * x) + 1.0
    if problem == 6:
        return 1.0 / (1.0 + x**2)


def evaluate_solution(problem: int, c: np.ndarray, x: np.ndarray) -> np.ndarray:
    if problem == 5:
        return clenshaw(c, x)
    if problem == 6:
        t = x / 2.5 - 1.0
        b, _ = boundary_lift(t)
        return clenshaw(c, t) + b

def solve_problem5(N: int):
    """Return nodes x, solved values U, and coefficients c for u_N."""
    x, D = chebyshev_matrix(N)
    D2 = D @ D
    D4 = D2 @ D2
    A = D4 - 4.0 * D2 + 3.0 * np.eye(N)

    g = (np.pi**4 + 4.0*np.pi**2 + 3.0) * np.cos(np.pi*x) + 3.0
    U = solve_zero_boundary(A, D, g)
    c = chebyshev_coefficients(U)
    return x, U, c

def solve_problem6(N: int):
    """Return nodes t, solved values V, and coefficients c for v_N."""
    t, D = chebyshev_matrix(N)
    x = 2.5 * (t + 1.0)
    alpha = 2.0 / 5.0
    D2 = D @ D
    D4 = D2 @ D2
    A = alpha**4 * D4 + alpha * D + np.eye(N)

    g = ((24.0 - 240.0*x**2 + 120.0*x**4) / (1.0 + x**2)**5
         + 1.0 / (1.0 + x**2) - 2.0*x / (1.0 + x**2)**2)

    b, db = boundary_lift(t)
    rhs = g - alpha * db - b
    V = solve_zero_boundary(A, D, rhs)
    c = chebyshev_coefficients(V)
    return t, V, c


def run_experiment(problem: int, N_to_test: List[int], N_to_plot: List[int]):
    if problem == 5:
        solver, bounds = solve_problem5, (-1.0, 1.0)
    elif problem == 6:
        solver, bounds = solve_problem6, (0.0, 5.0)

    x = np.linspace(bounds[0], bounds[1], 4001)
    exact = exact_solution(problem, x)
    errors = []

    fig_u, ax_u = plt.subplots()
    fig_e, ax_e = plt.subplots()
    ax_u.plot(x, exact, label="Exact", linewidth=2)

    for N in N_to_test:
        _, _, c = solver(N)
        numerical = evaluate_solution(problem, c, x)
        error = np.abs(numerical - exact)
        errors.append(np.max(error))   # No division by scale.

        if N in N_to_plot:
            ax_u.plot(x, numerical, "--", label=f"N = {N}")
            ax_e.semilogy(x, np.ma.masked_equal(error, 0), label=f"N = {N}")

    eps = np.finfo(float).eps

    for ax, title, xlabel, ylabel in [
        (ax_u, "Solutions", "x", "u(x)"),
        (ax_e, "Pointwise error", "x", "Absolute error"),
    ]:
        ax.set(title=f"Problem {problem}: {title}", xlabel=xlabel, ylabel=ylabel)
        ax.legend()
        ax.grid(True)
        ax.figure.tight_layout()

    best = np.argmin(errors)
    print(f"Problem {problem}: best tested N = {N_to_test[best]}, "
          f"Max pointwise error = {errors[best]:.3e}")
    hits = [N for N, error in zip(N_to_test, errors) if error <= eps]
    print("First tested N reaching epsilon:", hits[0] if hits else "none")

    output_dir = Path("figs") / f"problem_{problem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_u.savefig(output_dir / "solution.png", dpi=300, bbox_inches="tight")
    fig_e.savefig(output_dir / "pointwise_error.png", dpi=300, bbox_inches="tight")

def main():
    run_experiment(5, range(5, 65), [6, 10, 16, 21])
    run_experiment(6, range(5, 101), [8, 20, 40, 61])
    plt.show()

if __name__ == "__main__":
    main()