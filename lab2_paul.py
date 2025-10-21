import math
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import numdifftools as nd

eps = 10**-4
maxit = 1500


def f(x):
    return 1 / 8 * (x[0] ** 2 * x[1] + x[0] * x[1] ** 2 - x[0] * x[1])


# def f(x):
#     return - 1/8 * x[0] * x[1] * x[2]


def f_gradient(x):
    return [1 / 8 * (2 * x[0] * x[1] + x[1] ** 2 - x[1]), 1 / 8 * (x[0] ** 2 + 2 * x[0] * x[1] - x[0])]


# one gradient descent iteration
def gradient_iteration(f, x, y, gamma, grad):
    x = x - gamma * grad[0]
    y = y - gamma * grad[1]
    z = f([x, y])
    return x, y, z


# # gradient descent
# def gradient_descent(f, f_gradient, X0, gamma, eps, maxit):
#     X = [[X0[0], X0[1], f([X0[0], X0[1]])]]
#     gradient_calls = 0
#     func_calls = 1

#     while True:
#         grad = f_gradient(X[-1])
#         gradient_calls += 1
#         if grad == [0,0]:
#             break

#         x, y, z = gradient_iteration(f, X[-1][0], X[-1][1], gamma, grad)
#         func_calls += 1
#         X.append([x, y, z])
#         if linalg.norm(grad) < eps or len(X) >= maxit:
#             break

#     return X[-1], X, gradient_calls, func_calls

# def gradient_iteration(x, y, z, gamma, grad):
#     x = x - gamma * grad[0]
#     y = y - gamma * grad[1]
#     z = z - gamma * grad[2]
#     return x, y, z


# gradient descent
def gradient_descent(f, f_gradient, X0, eps, maxit):
    X = [[X0[0], X0[1], f([X0[0], X0[1]])]]
    gradient_calls = 0
    func_calls = 1
    gamma = 2  # Or pass this as a parameter

    while True:
        grad = f_gradient(X[-1])
        gradient_calls += 1
        if grad == [0, 0]:
            break

        x, y, z = gradient_iteration(f, X[-1][0], X[-1][1], gamma, grad)
        func_calls += 1
        X.append([x, y, z])
        if linalg.norm(grad) < eps or len(X) >= maxit:
            break

    return X[-1], X, gradient_calls, func_calls


# steepest descent
def steepest_descent(f, f_gradient, X0, eps, maxit):
    X = [[X0[0], X0[1], f([X0[0], X0[1]])]]
    gradient_calls = 0
    func_calls = 1
    while True:
        grad = f_gradient(X[-1])
        gradient_calls += 1

        if grad == [0, 0]:
            break

        gamma, fn_calls = split_half_method_for_fastest_descent(f, X[-1], grad, 0, 4, 0.0001)
        func_calls += fn_calls
        x, y, z = gradient_iteration(f, X[-1][0], X[-1][1], gamma, grad)
        func_calls += 1
        X.append([x, y, z])

        if linalg.norm(grad) < eps or len(X) >= maxit:
            break

    return X[-1], X, gradient_calls, func_calls


def split_half_method_for_fastest_descent(f, Xi, grad, l, r, eps):
    calls = 0

    def func(x):
        return f([Xi[0] - x * grad[0], Xi[1] - x * grad[1]])

    xm = (l + r) / 2
    fm = func(xm)
    calls += 1
    L = r - l
    while L >= eps:
        x1 = l + L / 4
        f1 = func(x1)
        calls += 1
        if f1 < fm:
            r = xm
            xm = x1
            fm = f1
        else:
            x2 = r - L / 4
            f2 = func(x2)
            calls += 1
            if f2 < fm:
                l = xm
                xm = x2
                fm = f2
            else:
                l = x1
                r = x2
        L = r - l
    return xm, calls


# deformed simplex algorithm
def deformed_simplex(f, X0, n, eps, alpha, beta, gamma, niu, maxit):
    def sort_simplex(X):
        return sorted(X, key=lambda x: x[-1])

    def deform_simplex(midpoint, reflection, theta):
        deformed = [0] * (n + 1)
        for i in range(n):
            deformed[i] = midpoint[i] + theta * (reflection[i] - midpoint[i])
        deformed[n] = f(deformed)
        return deformed

    func_calls = 1
    checkpoints = []
    delta_1 = (math.sqrt(n + 1) + n - 1) / (n * math.sqrt(2)) * alpha
    delta_2 = (math.sqrt(n + 1) - 1) / (n * math.sqrt(2)) * alpha
    X = [[0] * (n + 1) for _ in range(n + 1)]
    X[0] = [X0[0], X0[1], f(X0)]

    for i in range(1, n + 1):
        for j in range(0, n):
            if i != j + 1:
                X[i][j] = X0[j] + delta_1
            else:
                X[i][j] = X0[j] + delta_2
        X[i][n] = f(X[i])
        func_calls += 1
    checkpoints.append(X)

    while math.sqrt(sum((X[n][i] - X[0][i]) ** 2 for i in range(n))) >= eps and len(X) <= maxit:
        X = sort_simplex(X)

        midpoint = [0] * (n + 1)
        for i in range(0, n):
            midpoint[i] = (X[0][i] + X[n - 1][i]) / 2
        reflection = [0] * (n + 1)

        for i in range(0, n):
            reflection[i] = midpoint[i] + (midpoint[i] - X[n][i])
        reflection[n] = f(reflection)
        func_calls += 1

        if reflection[n] < X[0][n]:
            deformed = [0] * (n + 1)
            deformed = deform_simplex(midpoint, reflection, gamma)
            func_calls += 1
            if deformed[n] < reflection[n]:
                X[n] = deformed
            else:
                X[n] = reflection
        elif reflection[n] < X[n - 1][n]:
            X[n] = reflection
        elif reflection[n] < X[n][n]:
            deformed = [0] * (n + 1)
            deformed = deform_simplex(midpoint, reflection, beta)
            func_calls += 1
            if deformed[n] < reflection[n]:
                X[n] = deformed
            else:
                X[n] = reflection
        else:
            deformed = [0] * (n + 1)
            deformed = deform_simplex(midpoint, reflection, niu)
            func_calls += 1
            X[n] = deformed

        while any(X[n][i] < 0 for i in range(n)):
            deformed = [0] * (n + 1)
            deformed = deform_simplex(midpoint, reflection, niu)
            func_calls += 1
            X[n] = deformed

        checkpoints.append(X)
    X = sort_simplex(X)

    return X[0], checkpoints, func_calls


# Tables


def plot_figures(data, isSimplex):
    l = -0.1  # interval start
    r = 2  # interval end
    x = np.linspace(l, r, 1000)
    y = np.linspace(l, r, 1000)
    z = f([x, y])
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=100, alpha=0.5)
    plt.colorbar(label="z")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis("equal")
    if isSimplex:
        for triangle in data:
            triangle.append(triangle[0])
            triangle = np.array(triangle)
            plt.plot(triangle[:, 0], triangle[:, 1], "r-", linewidth=1)
    else:
        x_values, y_values, z_values = zip(*data)
        plt.scatter(x_values, y_values, label="Checkpoints", marker=".")
        plt.scatter(grad_points[-1][0], grad_points[-1][1], label="mininum", marker="d", color="red")
    plt.show()


def plot_points(plt, x_values, y_values):
    plt.scatter(x_values, y_values, label="Checkpoints", marker=".")
    plt.scatter(
        grad_points[-1][0], grad_points[-1][1], label="mininum", marker="d", color="red"
    )  # rezultato atvaizdavimas


def plot_triangles(data):
    for triangle in data:
        triangle.append(triangle[0])
        triangle = np.array(triangle)
        plt.plot(triangle[:, 0], triangle[:, 1], "r-", linewidth=1)


# grad_result, grad_points, grad_gradient_calls, grad_func_calls = gradient_descent(f, f_gradient, [0, 0], 2, eps, maxit)
# sd_result, sd_points, sd_gradient_calls, sd_func_calls = steepest_descent(f, f_gradient, [0, 0], eps, maxit)
# simplex_result, simplex_checkpoints, simplex_func_calls = deformed_simplex(f, [0, 0], 2, eps, 1, 0.5, 2, -0.5, maxit)

# tab = PrettyTable(["+++++++++++++++++++++++++", "Iteraciju sk.", "f(x) kvietimu sk.", "Gradiento skaic. sk.", "min X", "min f(X)"])
# tab.align = "l"
# tab.add_rows([
#     ["Gradientinis nusileidimas", len(grad_points), grad_func_calls, grad_gradient_calls, f"({grad_result[0]}, {grad_result[1]})", grad_result[2]],
#     ["Greiciausias nusileidimas", len(sd_points), sd_func_calls, sd_gradient_calls, f"({sd_result[0]}, {sd_result[1]})", sd_result[2]],
#     ["Deformuotas simpleksas", len(simplex_checkpoints), simplex_func_calls, "0", f"({simplex_result[0]}, {simplex_result[1]})", simplex_result[2]]
# ])

# print(tab)
# tab.clear_rows()

# # plot_figures(grad_points, False)
# # plot_figures(sd_points, False)
# # plot_figures(simplex_checkpoints, True)

# grad_result, grad_points, grad_gradient_calls, grad_func_calls = gradient_descent(f, f_gradient, [1, 1, 1], 2, eps, maxit)
# i = 1
# print("Rezultatai gradientinio nusileidimo metodu")
# print("------------------------------------------")
# for p in grad_points:
#     if i == 1 or i % 5 == 0 or i == len(grad_points):
#         print("Po", i, "iteraciju, gauti rezultatai:", p)
#     i += 1
# sd_result, sd_points, sd_gradient_calls, sd_func_calls = steepest_descent(f, f_gradient, [1, 1], eps, maxit)
# print("")
# print("Rezultatai greiciausio nusileidimo metodu")
# print("------------------------------------------")
# i = 1
# for p in sd_points:
#     if i == 1 or i % 5 == 0 or i == len(sd_points):
#         print("Po", i, "iteraciju, gauti rezultatai:", p)
#     i += 1
# simplex_result, simplex_checkpoints, simplex_func_calls = deformed_simplex(f, [1, 1], 2, eps, 1, 0.5, 2, -0.5, maxit)
# print("")
# print("Rezultatai simplekso metodu(tasku koordinates)")
# print("------------------------------------------")
# i = 1
# for p in simplex_checkpoints:
#     if i == 1 or i % 5 == 0 or i == len(simplex_checkpoints):
#         print("Po", i, "iteraciju, gauti rezultatai:")
#         print("                      ", p[0])
#         print("                      ", p[1])
#         print("                      ", p[2])
#     i += 1

# tab.add_rows([
#     ["Gradientinis nusileidimas", len(grad_points), grad_func_calls, grad_gradient_calls, f"({grad_result[0]}, {grad_result[1]})", grad_result[2]],
#     ["Greiciausias nusileidimas", len(sd_points), sd_func_calls, sd_gradient_calls, f"({sd_result[0]}, {sd_result[1]})", sd_result[2]],
#     ["Deformuotas simpleksas", len(simplex_checkpoints), simplex_func_calls, "0", f"({simplex_result[0]}, {simplex_result[1]})", simplex_result[2]]
# ])

# print(tab)
# tab.clear_rows()

# plot_figures(grad_points, False)
# plot_figures(sd_points, False)
# plot_figures(simplex_checkpoints, True)

grad_result, grad_points, grad_gradient_calls, grad_func_calls = gradient_descent(f, f_gradient, [0.1, 0.7], eps, maxit)
i = 1
print("Rezultatai gradientinio nusileidimo metodu")
print("------------------------------------------")
for p in grad_points:
    if i == 1 or i % 5 == 0 or i == len(grad_points):
        print("Po", i, "iteraciju, gauti rezultatai:", p)
    i += 1
sd_result, sd_points, sd_gradient_calls, sd_func_calls = steepest_descent(f, f_gradient, [0.1, 0.7], eps, maxit)
print("")
print("Rezultatai greiciausio nusileidimo metodu")
print("------------------------------------------")
i = 1
for p in sd_points:
    if i == 1 or i % 5 == 0 or i == len(sd_points):
        print("Po", i, "iteraciju, gauti rezultatai:", p)
    i += 1
simplex_result, simplex_checkpoints, simplex_func_calls = deformed_simplex(
    f, [0.1, 0.7], 2, eps, 1, 0.5, 2, -0.5, maxit
)
print("")
print("Rezultatai simplekso metodu(tasku koordinates)")
print("------------------------------------------")
i = 1
for p in simplex_checkpoints:
    if i == 1 or i % 5 == 0 or i == len(simplex_checkpoints):
        print("Po", i, "iteraciju, gauti rezultatai:")
        print("                      ", p[0])
        print("                      ", p[1])
        print("                      ", p[2])
    i += 1

# tab.add_rows([
#     ["Gradientinis nusileidimas", len(grad_points), grad_func_calls, grad_gradient_calls, f"({grad_result[0]}, {grad_result[1]})", grad_result[2]],
#     ["Greiciausias nusileidimas", len(sd_points), sd_func_calls, sd_gradient_calls, f"({sd_result[0]}, {sd_result[1]})", sd_result[2]],
#     ["Deformuotas simpleksas", len(simplex_checkpoints), simplex_func_calls, "0", f"({simplex_result[0]}, {simplex_result[1]})", simplex_result[2]]
# ])

# print(tab)

# # plot_figures(grad_points, False)
# # plot_figures(sd_points, False)
# plot_figures(simplex_checkpoints, True)
