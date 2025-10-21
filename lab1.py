# Hubertas Vindzigalskis, 1 lab.
import math
from prettytable import PrettyTable
from tabulate import tabulate

# Auksinis santykis
tau = (math.sqrt(5) - 1) / 2

# Du paskutiniai studento ID skaitmenys
a = 1
b = 7


# Optimizuojamoji funkcija
def f(x):
    return (x**2 - a) ** 2 / b - 1


# Pirmosios eiles isvestine
def df(x):
    return (2 * x * (x**2 - 1)) / 3


# Antrosios eiles isvestine
def ddf(x):
    return (2 * (3 * x**2) - 1) / 3


# Intervalo dalijimo pusiau metodas
def run_split_half(left, right, len):
    L = right - left
    iter = 0
    runs = 1
    x1 = left + L / 4
    x2 = right - L / 4
    xm = (left + right) / 2
    fxm = f(xm)

    steps = [[iter, f"[{left:.8f}, {right:.8f}]", x1, xm, x2, f(x1), fxm, f(x2), L]]

    while L > len:
        xm = (left + right) / 2
        x1 = left + L / 4
        x2 = right - L / 4

        fx1 = f(x1)
        fx2 = f(x2)
        runs += 2

        if fx1 < fxm:
            right = xm
            xm = x1
            fxm = fx1
        elif fx2 < fxm:
            left = xm
            xm = x2
            fxm = fx2
        else:
            left = x1
            right = x2

        L = right - left
        iter += 1

        steps.append([iter, f"[{left:.8f}, {right:.8f}]", x1, xm, x2, fx1, fxm, fx2, L])

    print("Itervalo dalijimo pusiau metodas:")
    print(
        tabulate(
            steps,
            headers=["Iteracija", "Rėžis[l, r]", "x_1", "x_m", "x_2", "f(x_1)", "f(x_m)", "f(x_2)", "L"],
            floatfmt=(None, None, ".8f", ".8f", ".8f", ".8f", ".8f", ".8f", ".12f"),
            numalign="left",
        )
    )
    return (xm, iter, runs)


# AUksinio pjuvio metodas
def run_golden_split(left, right, len):
    L = right - left
    iter = 0
    execs = 2
    xm = (left + right) / 2
    x1 = right - tau * L
    x2 = left + tau * L

    fx1 = f(x1)
    fx2 = f(x2)

    steps = [[iter, f"[{left:.8f}, {right:.8f}]", x1, xm, x2, fx1, f(xm), fx2, L]]

    while L > len:
        if fx2 < fx1:
            left = x1
            x1 = x2
            fx1 = fx2
            L = right - left
            x2 = left + tau * L
            fx2 = f(x2)
        else:
            right = x2
            x2 = x1
            fx2 = fx1
            L = right - left
            x1 = right - tau * L
            fx1 = f(x1)
        execs += 1

        iter += 1

        xm = (left + right) / 2
        steps.append([iter, f"[{left:.8f}, {right:.8f}]", x1, xm, x2, fx1, f(xm), fx2, L])

    print("Auksinio pjuvio metodas:")
    print(
        tabulate(
            steps,
            headers=["Iteracija", "Rėžis[l, r]", "x_1", "x_m", "x_2", "f(x_1)", "f(x_m)", "f(x_2)", "L"],
            floatfmt=(None, None, ".8f", ".8f", ".8f", ".8f", ".8f", ".8f", ".12f"),
            numalign="left",
        )
    )
    return (xm, iter, execs)


# Niutono metodas
def run_newton(x0, len):
    xi = x0 - df(x0) / ddf(x0)
    iter = 0

    steps = [[iter, x0, xi, f(xi), abs(x0 - xi)]]

    while abs(x0 - xi) > len and iter <= 100:
        x0 = xi
        xi = x0 - df(x0) / ddf(x0)
        iter += 1
        steps.append([iter, x0, xi, f(xi), abs(x0 - xi)])

    print("Niutono metodas:")
    print(
        tabulate(
            steps,
            headers=["Iteracija", "x_i-1", "x_i", "f(x_i)", "|x_i-1 - x_i|"],
            floatfmt=(None, ".8f", ".8f", ".8f", ".12f"),
            numalign="left",
        )
    )
    return (xi, iter, iter)


(split, s_iter, s_execs) = run_split_half(0, 10, 10**-4)
print()
(gss, g_iter, g_execs) = run_golden_split(0, 10, 10**-4)
print()
(newton, n_iter, n_execs) = run_newton(5, 10**-4)
print()


results = [
    ["x_min", split, gss, newton],
    ["f(x_min)", f(split), f(gss), f(newton)],
    ["Iteracijų sk.", s_iter, g_iter, n_iter],
    ["f(x) iškvietimu sk.", s_execs, g_execs, "-"],
    ["f'(x) iškvietimu sk.", "-", "-", n_execs],
    ["f''(x) iškvietimu sk.", "-", "-", n_execs],
]
print(
    tabulate(
        results,
        headers=["Rezultato pavadinimas", "Intervalo dalijimas pusiau", "Auksinis pjūvis", "Niutono metodas"],
        floatfmt=(None, ".8f", ".12f", ".12f", ".12f"),
        numalign="left",
    )
)
