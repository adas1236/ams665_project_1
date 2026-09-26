import numpy as np

def boundary_lift(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)

    p0 = (t - 1.0)**2 * (t + 2.0) / 4.0
    p1 = (t + 1.0)**2 * (2.0 - t) / 4.0
    q1 = (t + 1.0)**2 * (t - 1.0) / 4.0
    dp0 = 3.0 * (t**2 - 1.0) / 4.0
    dp1 = 3.0 * (1.0 - t**2) / 4.0
    dq1 = (3.0*t**2 + 2.0*t - 1.0) / 4.0

    slope_right_t = (5.0 / 2.0) * (-10.0 / 676.0)
    b = p0 + p1 / 26.0 + slope_right_t * q1
    db = dp0 + dp1 / 26.0 + slope_right_t * dq1
    return b, db