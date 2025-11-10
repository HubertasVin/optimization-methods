import math
import numpy as np
from prettytable import PrettyTable

# ============================================================================
# PARAMETRAI
# ============================================================================
e = 0.0001
maxit = 20  # Maksimalus išorinių iteracijų skaičius

# Studento knygelės numeris: 2*1*abc
a, b, c = 1, 7, 0

points = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (a / 10, b / 10, c / 10 if c != 0 else 0.1)]

# Skirtingos r konfigūracijos testavimui
r_configs = [
    {"r_start": 10, "r_mult": 0.5, "name": "A (r=10, n=0.5)"},
    {"r_start": 5, "r_mult": 0.3, "name": "B (r=5, n=0.3)"},
    {"r_start": 2, "r_mult": 0.1, "name": "C (r=2, n=0.1)"},
]

# ============================================================================
# PAGALBINĖS FUNKCIJOS
# ============================================================================


def distance_beetween_vectors(X1, X2):
    """Atstumas tarp dviejų vektorių"""
    return np.linalg.norm(np.array(X1) - np.array(X2))


def simplex_method(objective_function, X0, r):
    """
    3D Nelder-Mead simplex metodas.
    OPTIMIZUOTAS: mažesnis max_iter ir tolerance
    """
    n = len(X0)
    alpha_size = 1.0
    alpha = 1.0
    beta = 0.5
    gamma = 2.0
    niu = -0.5
    tolerance = 1e-5  # Sumažintas iš 1e-6
    max_iter = 200  # Sumažintas iš 500

    def f_penalty(X):
        return objective_function(X, r)

    def sort_simplex(X):
        return sorted(X, key=lambda x: x[-1])

    def deform_simplex(midpoint, reflection, theta):
        deformed = [0] * (n + 1)
        for i in range(n):
            deformed[i] = midpoint[i] + theta * (reflection[i] - midpoint[i])
        deformed[n] = f_penalty(deformed[:n])
        return deformed

    delta_1 = (math.sqrt(n + 1) + n - 1) / (n * math.sqrt(2)) * alpha_size
    delta_2 = (math.sqrt(n + 1) - 1) / (n * math.sqrt(2)) * alpha_size

    X = [[0] * (n + 1) for _ in range(n + 1)]
    X[0] = list(X0) + [f_penalty(X0)]

    for i in range(1, n + 1):
        for j in range(n):
            if i != j + 1:
                X[i][j] = X0[j] + delta_1
            else:
                X[i][j] = X0[j] + delta_2
        X[i][n] = f_penalty(X[i][:n])

    for iteration in range(max_iter):
        X = sort_simplex(X)

        if math.sqrt(sum((X[n][i] - X[0][i]) ** 2 for i in range(n))) < tolerance:
            return (np.array(X[0][:n]), X[0][n], iteration + 1)

        midpoint = [0] * (n + 1)
        for i in range(n):
            midpoint[i] = (X[0][i] + X[n - 1][i]) / 2

        reflection = [0] * (n + 1)
        for i in range(n):
            reflection[i] = midpoint[i] + alpha * (midpoint[i] - X[n][i])
        reflection[n] = f_penalty(reflection[:n])

        if reflection[n] < X[0][n]:
            deformed = deform_simplex(midpoint, reflection, gamma)
            if deformed[n] < reflection[n]:
                X[n] = deformed
            else:
                X[n] = reflection
        elif reflection[n] < X[n - 1][n]:
            X[n] = reflection
        elif reflection[n] < X[n][n]:
            deformed = deform_simplex(midpoint, reflection, beta)
            if deformed[n] < reflection[n]:
                X[n] = deformed
            else:
                X[n] = reflection
        else:
            deformed = deform_simplex(midpoint, reflection, niu)
            X[n] = deformed

        # Projekcija į teigiamą ortantą
        while any(X[n][i] < 0 for i in range(n)):
            deformed = deform_simplex(midpoint, reflection, niu)
            X[n] = deformed

    X = sort_simplex(X)
    return (np.array(X[0][:n]), X[0][n], max_iter)


# ============================================================================
# TIKSLO IR APRIBOJIMŲ FUNKCIJOS
# ============================================================================


def f(X):
    """Tikslo funkcija: -V"""
    return -X[0] * X[1] * X[2]


def g(X):
    """Lygybinis apribojimas"""
    return 2 * (X[0] * X[1] + X[0] * X[2] + X[1] * X[2]) - 1


def h(x):
    """Nelygybinis apribojimas"""
    return -x


def penalty_function(X):
    """Baudos narys"""
    return pow(max(0, h(X[0])), 2) + pow(max(0, h(X[1])), 2) + pow(max(0, h(X[2])), 2) + pow(g(X), 2)


def objective_function(X, r=2):
    """Baudos funkcija B(X,r)"""
    return f(X) + (1 / r) * penalty_function(X)


def minimize(X0, e, maxit, r, r_multiplier):
    """Minimizuoja su baudos funkcija"""
    XN = simplex_method(objective_function, X0, r)
    count = XN[2]
    it = 1

    while distance_beetween_vectors(XN[0], X0) > e and it <= maxit:
        r *= r_multiplier
        X0 = XN[0]
        XN = simplex_method(objective_function, X0, r)
        count += XN[2]
        it += 1

    return XN[0], XN[1], it, count


# ============================================================================
# BAUDOS DAUGIKLIO ĮTAKOS TYRIMAS
# ============================================================================

print("BAUDOS DAUGIKLIO r ĮTAKOS TYRIMAS")
print("-" * 80)

test_point = np.array([0.5, 0.5, 0.5])
r_test = [10, 5, 1, 0.5, 0.1, 0.01]

print(f"Testuojamas taškas: X = ({test_point[0]:.2f}, {test_point[1]:.2f}, {test_point[2]:.2f})")
print(f"g(X) = {g(test_point):.6f} (pažeistas!)\n")

tab_penalty = PrettyTable(["r", "f(X)", "Baudos narys", "B(X,r)"])
tab_penalty.align = "r"

for r in r_test:
    B_val = objective_function(test_point, r)
    f_val = f(test_point)
    penalty = B_val - f_val

    tab_penalty.add_row([f"{r:.2f}", f"{f_val:.6f}", f"{penalty:.6f}", f"{B_val:.6f}"])

print(tab_penalty)
print("\nMažėjant r → baudos narys didėja → apribojimai griežčiau verčiami\n")

# ============================================================================
# OPTIMIZAVIMAS SU KELIOMIS r KONFIGŪRACIJOMIS
# ============================================================================

print("=" * 100)
print("OPTIMIZAVIMO REZULTATAI SU SKIRTINGOMIS r KONFIGŪRACIJOMIS")
print("=" * 100 + "\n")

all_results = []

for config in r_configs:
    r_start = config["r_start"]
    r_mult = config["r_mult"]
    config_name = config["name"]

    print(f"\nKONFIGŪRACIJA: {config_name}")
    print("-" * 100)

    tab_results = PrettyTable(["Pradinis taškas", "Sprendinys X*", "Tūris V", "|g(X*)|", "Iteracijos"])
    tab_results.align["Sprendinys X*"] = "l"
    tab_results.align["Tūris V"] = "r"

    config_results = []

    for i, point in enumerate(points):
        point_name = ["X₀=(0,0,0)", "X₁=(1,1,1)", f"Xₘ=({a/10},{b/10},{c/10 if c!=0 else 0.1})"][i]

        X_opt, f_min, outer_iter, total_iter = minimize(np.array(point), e, maxit, r_start, r_mult)

        V_opt = -f_min
        g_opt = g(X_opt)

        tab_results.add_row(
            [
                point_name,
                f"({X_opt[0]:.6f}, {X_opt[1]:.6f}, {X_opt[2]:.6f})",
                f"{V_opt:.8f}",
                f"{abs(g_opt):.2e}",
                total_iter,
            ]
        )

        config_results.append(
            {"config": config_name, "point": point_name, "X": X_opt, "V": V_opt, "g": g_opt, "iter": total_iter}
        )

    print(tab_results)
    all_results.extend(config_results)
