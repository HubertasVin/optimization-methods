import math
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

tau = (math.sqrt(5) - 1) / 2
a = 1
b = 7


def f(X):
    X1, X2 = X[0], X[1]
    return 1 / 8 * (X1**2 * X2 + X1 * X2**2 - X1 * X2)


def gradient_f(X):
    X1, X2 = X[0], X[1]
    df_dX1 = 1 / 8 * (2 * X1 * X2 + X2**2 - X2)
    df_dX2 = 1 / 8 * (X1**2 + 2 * X1 * X2 - X1)
    return np.array([df_dX1, df_dX2])


optimal_point = np.array([1 / 3, 1 / 3])
optimal_value = f(optimal_point)

X0 = np.array([0.0, 0.0])
X1 = np.array([1.0, 1.0])
Xm = np.array([a / 10, b / 10])
test_points = [("X0", X0), ("X1", X1), ("Xm", Xm)]

print("=" * 93)
results_table = []

for name, point in test_points:
    f_val = f(point)
    grad = gradient_f(point)
    X3 = 1 - point[0] - point[1]
    f_str = f"{f_val:.5f}"
    grad_str = f"[{grad[0]:.5f}, {grad[1]:.5f}]"
    results_table.append(
        [name, f"({point[0]:.1f}, {point[1]:.1f})", f"{X3:.2f}", f_str, grad_str, f"{np.linalg.norm(grad):.4f}"]
    )

print(tabulate(results_table, headers=["Taškas", "X=(X₁,X₂)", "X₃", "f(X)", "∇f(X)", "||∇f||"], numalign="left"))
print()


def gradient_descent(start_point, point_name="", max_iter=1000, tolerance=1e-6):
    X = np.copy(start_point)
    gamma = 3.0
    trajectory = [np.copy(X)]

    for i in range(max_iter):
        grad = gradient_f(X)
        grad_norm = np.linalg.norm(grad)

        print(f"[{point_name}] Epocha {i}: ||∇f|| = {grad_norm:.6f}")

        if grad_norm < tolerance:
            return X, i + 1, trajectory

        X = X - gamma * grad
        trajectory.append(np.copy(X))

    return X, max_iter, trajectory


def steepest_descent(start_point, point_name="", max_iter=1000, tolerance=1e-6):
    X = np.copy(start_point)
    trajectory = [np.copy(X)]

    def golden_section_search(X, grad, left=0, right=6.0, eps_len=0.0001):
        L = right - left

        def func_lambda(lam):
            return f(X - lam * grad)

        x1 = right - tau * L
        x2 = left + tau * L

        fx1 = func_lambda(x1)
        fx2 = func_lambda(x2)

        while L > eps_len:
            if fx2 < fx1:
                left = x1
                x1 = x2
                fx1 = fx2
                L = right - left
                x2 = left + tau * L
                fx2 = func_lambda(x2)
            else:
                right = x2
                x2 = x1
                fx2 = fx1
                L = right - left
                x1 = right - tau * L
                fx1 = func_lambda(x1)

        xm = (left + right) / 2
        return xm

    for i in range(max_iter):
        grad = gradient_f(X)
        grad_norm = np.linalg.norm(grad)

        if grad_norm < tolerance:
            return X, i + 1, trajectory

        step_size = golden_section_search(X, grad)

        print(f"[{point_name}] Epocha {i}: λ = {step_size:.6f}, ||∇f|| = {grad_norm:.6f}")

        X = X - step_size * grad
        trajectory.append(np.copy(X))

    return X, max_iter, trajectory


def deformable_simplex(start_point, point_name="", max_iter=1000, tolerance=1e-6):
    n = 2
    alpha_size = 1.0
    alpha = 1.0
    beta = 0.5
    gamma = 2.0
    niu = -0.5
    simplex_history = []

    def sort_simplex(X):
        return sorted(X, key=lambda x: x[-1])

    def deform_simplex(midpoint, reflection, theta):
        deformed = [0] * (n + 1)
        for i in range(n):
            deformed[i] = midpoint[i] + theta * (reflection[i] - midpoint[i])
        deformed[n] = f(deformed[:n])
        return deformed

    delta_1 = (math.sqrt(n + 1) + n - 1) / (n * math.sqrt(2)) * alpha_size
    delta_2 = (math.sqrt(n + 1) - 1) / (n * math.sqrt(2)) * alpha_size

    X = [[0] * (n + 1) for _ in range(n + 1)]
    X[0] = [start_point[0], start_point[1], f(start_point)]

    for i in range(1, n + 1):
        for j in range(n):
            if i != j + 1:
                X[i][j] = start_point[j] + delta_1
            else:
                X[i][j] = start_point[j] + delta_2
        X[i][n] = f(X[i][:n])

    simplex_history.append([[vertex[0], vertex[1]] for vertex in X])

    for iteration in range(max_iter):
        X = sort_simplex(X)

        if math.sqrt(sum((X[n][i] - X[0][i]) ** 2 for i in range(n))) < tolerance:
            return np.array(X[0][:n]), iteration + 1, simplex_history

        midpoint = [0] * (n + 1)
        for i in range(n):
            midpoint[i] = (X[0][i] + X[n - 1][i]) / 2

        reflection = [0] * (n + 1)
        for i in range(n):
            reflection[i] = midpoint[i] + alpha * (midpoint[i] - X[n][i])
        reflection[n] = f(reflection[:n])

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

        while any(X[n][i] < 0 for i in range(n)):
            deformed = deform_simplex(midpoint, reflection, niu)
            X[n] = deformed

        if iteration % 3 == 0 or iteration < 20:
            simplex_history.append([[vertex[0], vertex[1]] for vertex in X])

    X = sort_simplex(X)
    simplex_history.append([[vertex[0], vertex[1]] for vertex in X])
    return np.array(X[0][:n]), max_iter, simplex_history


def run_optimization_comparison():
    algorithms = [
        ("Gradientinis nusileidimas", gradient_descent),
        ("Greičiausias nusileidimas", steepest_descent),
        ("Deformuojamo simplekso algoritmas", deformable_simplex),
    ]

    starting_points = [("X0", X0), ("X1", X1), ("Xm", Xm)]

    print("=" * 93)
    print("OPTIMIZAVIMO REZULTATAI:")
    print("=" * 93)

    all_results = []
    all_trajectories = {}

    for alg_name, algorithm in algorithms:
        print(f"\n{alg_name}:")
        alg_results = []
        all_trajectories[alg_name] = {}

        for point_name, start_point in starting_points:
            print(f"\n--- Pradinis taškas: {point_name} ---")
            try:
                result_point, iterations, trajectory = algorithm(start_point, point_name, tolerance=1e-5)
                final_value = f(result_point)
                error = np.linalg.norm(result_point - optimal_point)
                X3 = 1 - result_point[0] - result_point[1]
                volume = result_point[0] * result_point[1] * X3

                alg_results.append(
                    [
                        point_name,
                        f"({result_point[0]:.10f}, {result_point[1]:.10f})",
                        f"{final_value:.10f}",
                        f"{volume:.10f}",
                        f"{error:.10f}",
                        iterations,
                    ]
                )

                all_trajectories[alg_name][point_name] = trajectory

            except Exception as e:
                alg_results.append([point_name, "Klaida", str(e), "-", "-", "-"])
                all_trajectories[alg_name][point_name] = None

        print("\n")
        print(
            tabulate(
                alg_results,
                headers=["Pradinis", "Sprendinys X*", "f(X*)", "V", "Paklaida", "Iter."],
                numalign="left",
                floatfmt=".10f",
            )
        )
        print("\n" + "=" * 93)

        all_results.extend(alg_results)

    return all_results, all_trajectories


def visualize_optimization_paths(trajectories):
    def plot_contours(ax, X1_grid, X2_grid, Z, levels):
        contour = ax.contour(X1_grid, X2_grid, Z, levels=levels, alpha=0.5, linewidths=0.8)
        ax.clabel(contour, inline=True, fontsize=9, fmt="%.4f")
        return contour

    def plot_basic_elements(ax, show_start=True):
        ax.plot(
            optimal_point[0],
            optimal_point[1],
            "r*",
            markersize=20,
            label="Optimumas",
            markeredgecolor="black",
            markeredgewidth=1,
        )
        if show_start:
            ax.plot(
                Xm[0], Xm[1], "ko", markersize=12, label="Pradžia (Xm)", markerfacecolor="yellow", markeredgewidth=2
            )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=10)

    def plot_trajectory(ax, traj, color, marker_size=4):
        traj_array = np.array(traj)
        ax.plot(
            traj_array[:, 0],
            traj_array[:, 1],
            color=color,
            alpha=0.7,
            linewidth=2.5,
            label=f"Kelias ({len(traj)} iter.)",
        )
        ax.plot(
            traj_array[:, 0],
            traj_array[:, 1],
            "o",
            color=color,
            markersize=marker_size,
            markeredgecolor="black",
            markeredgewidth=1.5,
        )
        ax.plot(
            traj_array[-1, 0],
            traj_array[-1, 1],
            "o",
            color=color,
            markersize=12,
            markeredgecolor="black",
            markeredgewidth=1.5,
        )

    def plot_simplex_vertices(ax, simplex, idx, total_simplexes):
        vertex_alpha = 0.4 + 0.6 * (idx / total_simplexes)
        for vertex in simplex:
            ax.plot(vertex[0], vertex[1], "o", color="red", markersize=4, alpha=vertex_alpha, zorder=5)
            ax.plot(
                vertex[0],
                vertex[1],
                "o",
                markerfacecolor="none",
                markeredgecolor="black",
                markersize=4,
                markeredgewidth=1.2,
                zorder=6,
            )

    x1_range = np.linspace(-0.2, 1.2, 100)
    x2_range = np.linspace(-0.2, 1.7, 100)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)

    Z = np.array([[f([X1_grid[j, i], X2_grid[j, i]]) for i in range(len(x1_range))] for j in range(len(x2_range))])
    levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Optimizavimo algoritmų palyginimas (pradinis taškas Xm)", fontsize=18, fontweight="bold")

    configs = [
        ("Bendras vaizdas - pradžios taškai", None, (0, 0)),
        ("Gradientinis nusileidimas", "Gradientinis nusileidimas", (0, 1)),
        ("Greičiausias nusileidimas", "Greičiausias nusileidimas", (1, 0)),
        ("Deformuojamo simplekso algoritmas", "Deformuojamo simplekso algoritmas", (1, 1)),
    ]

    for title, alg_name, (row, col) in configs:
        ax = axes[row, col]

        plot_contours(ax, X1_grid, X2_grid, Z, levels)
        ax.set_xlim([-0.2, 1.2])
        ax.set_ylim([-0.2, 1.7])

        if alg_name is None:
            plot_basic_elements(ax, show_start=False)
            colors = ["cyan", "magenta", "yellow"]
            for i, (name, point) in enumerate(test_points):
                ax.plot(
                    point[0],
                    point[1],
                    "o",
                    color=colors[i],
                    markersize=12,
                    label=name,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                )
        else:
            plot_basic_elements(ax)
            traj = trajectories[alg_name].get("Xm")

            if traj and alg_name != "Deformuojamo simplekso algoritmas":
                color = "blue" if "Gradientinis" in alg_name else "green"
                plot_trajectory(ax, traj, color)
            elif traj:
                total_simplexes = len(traj)

                for idx, simplex in enumerate(traj):
                    if len(simplex) >= 3:
                        triangle = np.array(simplex + [simplex[0]])
                        alpha_val = 0.3 + 0.5 * (idx / total_simplexes)
                        ax.plot(triangle[:, 0], triangle[:, 1], color="red", alpha=alpha_val, linewidth=2.5, zorder=3)
                        ax.fill(triangle[:, 0], triangle[:, 1], color="red", alpha=0.03, zorder=2)
                        plot_simplex_vertices(ax, simplex, idx, total_simplexes)

                if traj and traj[-1]:
                    final = traj[-1][0]
                    ax.plot(
                        final[0],
                        final[1],
                        "o",
                        color="red",
                        markersize=12,
                        label=f"Simpleksų sk.: {len(traj)}",
                        markeredgecolor="black",
                        markeredgewidth=1.5,
                        zorder=8,
                    )

        ax.set_xlabel("X₁", fontsize=12)
        ax.set_ylabel("X₂", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results, trajectories = run_optimization_comparison()
    visualize_optimization_paths(trajectories)
