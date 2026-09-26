import numpy as np

def chebyshev_matrix(N: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.cos(np.pi * np.arange(N) / (N - 1))
    a = np.empty(N)
    D = np.empty((N, N))

    for j in range(N):
        differences = t[j] - np.delete(t, j)
        a[j] = np.prod(differences) 
        D[j, j] = np.sum(1.0 / differences)

    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = a[i] / (a[j] * (t[i] - t[j]))
    return t, D

def chebyshev_coefficients(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = len(values) - 1
    theta = np.pi * np.arange(n + 1) / n

    f = values.copy()
    f[[0, -1]] *= 0.5
    c = np.empty(n + 1)
    for k in range(n + 1):
        c[k] = (2.0 / n) * np.sum(f * np.cos(k * theta))
    return c

def clenshaw(c: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    coeff = np.asarray(c, dtype=float).copy()

    coeff[[0, -1]] *= 0.5
    next1 = np.zeros_like(t)
    next2 = np.zeros_like(t)
    for k in range(len(coeff) - 1, 0, -1):
        current = coeff[k] + 2.0 * t * next1 - next2
        next2, next1 = next1, current
    return coeff[0] + t * next1 - next2


def solve_zero_boundary(A: np.ndarray, D: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    N = len(rhs)
    A, rhs = A.copy(), rhs.copy()
    I = np.eye(N)

    A[0, :] = I[0, :]
    A[1, :] = D[0, :]
    A[-2, :] = D[-1, :]
    A[-1, :] = I[-1, :]
    rhs[[0, 1, -2, -1]] = 0.0

    scale = np.max(np.abs(A), axis=1)
    return np.linalg.solve(A / scale[:, None], rhs / scale)