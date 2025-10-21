import math
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Auksinis santykis
tau = (math.sqrt(5) - 1) / 2

# Du paskutiniai studento ID skaitmenys
a = 1
b = 7


# Tikslo funkcija
def f(X):
    X1, X2 = X[0], X[1]
    return 1 / 8 * (X1**2 * X2 + X1 * X2**2 - X1 * X2)


# Gradiento funkcija
def gradient_f(X):
    X1, X2 = X[0], X[1]

    df_dX1 = 1 / 8 * (2 * X1 * X2 + X2**2 - X2)
    df_dX2 = 1 / 8 * (X1**2 + 2 * X1 * X2 - X1)

    return np.array([df_dX1, df_dX2])


# Optimaliausias taškas
optimal_point = np.array([1 / 3, 1 / 3])
optimal_value = f(optimal_point)


# Pradiniai taškai
X0 = np.array([0.0, 0.0])
X1 = np.array([1.0, 1.0])
Xm = np.array([a / 10, b / 10])
test_points = [("X0", X0), ("X1", X1), ("Xm", Xm)]

# Apskaičiuojame funkcijų reikšmes testuojamuose taškuose
print("=" * 93)
results_table = []

for name, point in test_points:
    f_val = f(point)
    grad = gradient_f(point)
    X3 = 1 - point[0] - point[1]

    # Formatuojame reikšmes
    f_str = f"{f_val:.5f}"
    grad_str = f"[{grad[0]:.5f}, {grad[1]:.5f}]"

    results_table.append(
        [name, f"({point[0]:.1f}, {point[1]:.1f})", f"{X3:.2f}", f_str, grad_str, f"{np.linalg.norm(grad):.4f}"]
    )

print(tabulate(results_table, headers=["Taškas", "X=(X₁,X₂)", "X₃", "f(X)", "∇f(X)", "||∇f||"], numalign="left"))
print()


def gradient_descent(start_point, max_iter=1000, tolerance=1e-6):
    X = np.copy(start_point)
    gamma = 3.0

    trajectory = [np.copy(X)]

    for i in range(max_iter):
        grad = gradient_f(X)

        if np.linalg.norm(grad) < tolerance:
            return X, i + 1, trajectory

        X = X - gamma * grad
        trajectory.append(np.copy(X))

    return X, max_iter, trajectory


def steepest_descent(start_point, max_iter=1000, tolerance=1e-6):
    X = np.copy(start_point)

    trajectory = [np.copy(X)]

    def golden_section_search(X, grad):
        left = 0
        right = 4.0
        eps_ls = 0.0001

        def func(gamma):
            return f(X - gamma * grad)

        while right - left > eps_ls:
            c = left + (1 - tau) * (right - left)
            d = left + tau * (right - left)

            fc = func(c)
            fd = func(d)

            if fc < fd:
                right = d
            else:
                left = c

        return (left + right) / 2

    for i in range(max_iter):
        grad = gradient_f(X)

        if np.linalg.norm(grad) < tolerance:
            return X, i + 1, trajectory

        step_size = golden_section_search(X, grad)

        X = X - step_size * grad
        trajectory.append(np.copy(X))

    return X, max_iter, trajectory


def deformable_simplex(start_point, max_iter=1000, tolerance=1e-6):
    n = 2

    alpha_size = 1.0  # Simplekso dydžio parametras
    alpha = 1.0  # Atspindžio koeficientas
    beta = 0.5  # Suspaudimo koeficientas
    gamma = 2.0  # Išplėtimo koeficientas
    niu = -0.5  # Vidinio suspaudimo koeficientas

    # Trajektorijos saugojimas - simpleksų sąrašas
    simplex_history = []

    def sort_simplex(X):
        return sorted(X, key=lambda x: x[-1])

    def deform_simplex(midpoint, reflection, theta):
        deformed = [0] * (n + 1)
        for i in range(n):
            deformed[i] = midpoint[i] + theta * (reflection[i] - midpoint[i])
        deformed[n] = f(deformed[:n])
        return deformed

    # Inicializuojame simpleksą kaip pavyzdyje - pradinis taškas yra viršūnė
    delta_1 = (math.sqrt(n + 1) + n - 1) / (n * math.sqrt(2)) * alpha_size
    delta_2 = (math.sqrt(n + 1) - 1) / (n * math.sqrt(2)) * alpha_size

    X = [[0] * (n + 1) for _ in range(n + 1)]
    X[0] = [start_point[0], start_point[1], f(start_point)]

    # Sukuriame pradines simplekso viršūnes
    for i in range(1, n + 1):
        for j in range(n):
            if i != j + 1:
                X[i][j] = start_point[j] + delta_1
            else:
                X[i][j] = start_point[j] + delta_2
        X[i][n] = f(X[i][:n])

    # Išsaugome pradinį simpleksą
    simplex_history.append([[vertex[0], vertex[1]] for vertex in X])

    for iteration in range(max_iter):
        X = sort_simplex(X)

        # Tikriname konvergavimą pagal simplekso dydį
        if math.sqrt(sum((X[n][i] - X[0][i]) ** 2 for i in range(n))) < tolerance:
            return np.array(X[0][:n]), iteration + 1, simplex_history

        # Apskaičiuojame vidurinį tašką - tik tarp geriausio ir antro blogiausio
        midpoint = [0] * (n + 1)
        for i in range(n):
            midpoint[i] = (X[0][i] + X[n - 1][i]) / 2

        # Atspindys
        reflection = [0] * (n + 1)
        for i in range(n):
            reflection[i] = midpoint[i] + alpha * (midpoint[i] - X[n][i])
        reflection[n] = f(reflection[:n])

        # Sprendimo logika pagal atspindžio rezultatą
        if reflection[n] < X[0][n]:
            # Bandome išplėtimą
            deformed = deform_simplex(midpoint, reflection, gamma)
            if deformed[n] < reflection[n]:
                X[n] = deformed
            else:
                X[n] = reflection

        elif reflection[n] < X[n - 1][n]:
            # Priimame atspindį
            X[n] = reflection

        elif reflection[n] < X[n][n]:
            # Išorinis suspaudimas
            deformed = deform_simplex(midpoint, reflection, beta)
            if deformed[n] < reflection[n]:
                X[n] = deformed
            else:
                X[n] = reflection
        else:
            # Vidinis suspaudimas
            deformed = deform_simplex(midpoint, reflection, niu)
            X[n] = deformed

        while any(X[n][i] < 0 for i in range(n)):
            deformed = deform_simplex(midpoint, reflection, niu)
            X[n] = deformed

        # Išsaugome simpleksą kas kelis žingsnius
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
            try:
                result_point, iterations, trajectory = algorithm(start_point, tolerance=1e-5)
                final_value = f(result_point)

                error = np.linalg.norm(result_point - optimal_point)

                # Apskaičiuojame tikrąjį tūrį iš rezultato
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

                # Išsaugome trajektoriją
                all_trajectories[alg_name][point_name] = trajectory

            except Exception as e:
                alg_results.append([point_name, "Klaida", str(e), "-", "-", "-"])
                all_trajectories[alg_name][point_name] = None

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


# ====== VIZUALIZACIJA ======


def visualize_optimization_paths(trajectories):
    # Pagalbinė funkcija konturų braižymui
    def plot_contours(ax, X1_grid, X2_grid, Z, levels):
        contour = ax.contour(X1_grid, X2_grid, Z, levels=levels, alpha=0.5, linewidths=0.8)
        ax.clabel(contour, inline=True, fontsize=9, fmt="%.4f")
        return contour

    # Pagalbinė funkcija pagrindinių elementų braižymui
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

    # Pagalbinė funkcija trajektorijos braižymui
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
            markeredgecolor=f"dark{color}",
            markeredgewidth=0.5,
        )
        ax.plot(
            traj_array[-1, 0],
            traj_array[-1, 1],
            "o",
            color=color,
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=1.5,
        )

    # Sukuriame tinklelį funkcijos vizualizacijai
    x1_edges, x2_edges = [-0.2, 1.1], [-0.2, 1.1]
    x1_range = np.linspace(x1_edges[0], x1_edges[1], 100)
    x2_range = np.linspace(x2_edges[0], x2_edges[1], 100)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)

    Z = np.array([[f([X1_grid[j, i], X2_grid[j, i]]) for i in range(len(x1_range))] for j in range(len(x2_range))])
    levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 70)

    # Sukuriame 2x2 grafikų tinklelį
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Optimizavimo algoritmų palyginimas (pradinis taškas Xm)", fontsize=18, fontweight="bold")

    # Konfigūracija kiekvienam grafikui
    configs = [
        ("Bendras vaizdas - pradžios taškai", None, (0, 0)),
        ("Gradientinis nusileidimas", "Gradientinis nusileidimas", (0, 1)),
        ("Greičiausias nusileidimas", "Greičiausias nusileidimas", (1, 0)),
        ("Deformuojamo simplekso algoritmas", "Deformuojamo simplekso algoritmas", (1, 1)),
    ]

    for title, alg_name, (row, col) in configs:
        ax = axes[row, col]

        # Simpleksui naudojame atitolinką vaizdą
        if alg_name == "Deformuojamo simplekso algoritmas":
            x1_s, x2_s = [-0.5, 1.2], [-0.5, 1.7]
            X1_s, X2_s = np.meshgrid(np.linspace(x1_s[0], x1_s[1], 100), np.linspace(x2_s[0], x2_s[1], 100))
            Z_s = np.array([[f([X1_s[j, i], X2_s[j, i]]) for i in range(100)] for j in range(100)])
            plot_contours(ax, X1_s, X2_s, Z_s, np.linspace(np.nanmin(Z_s), np.nanmax(Z_s), 80))
            ax.set_xlim(x1_s)
            ax.set_ylim(x2_s)
        else:
            plot_contours(ax, X1_grid, X2_grid, Z, levels)
            ax.set_xlim(x1_edges)
            ax.set_ylim(x2_edges)

        # Braižome pagrindinius elementus
        if alg_name is None:  # Bendras vaizdas
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
            elif traj:  # Simpleksas
                for idx, simplex in enumerate(traj):
                    if len(simplex) >= 3:
                        triangle = np.array(simplex + [simplex[0]])
                        alpha_val = 0.3 + 0.5 * (idx / len(traj))
                        ax.plot(triangle[:, 0], triangle[:, 1], color="red", alpha=alpha_val, linewidth=2)
                        ax.fill(triangle[:, 0], triangle[:, 1], color="red", alpha=0.03)

                if traj and traj[-1]:
                    ax.plot(
                        traj[-1][0][0],
                        traj[-1][0][1],
                        "o",
                        color="red",
                        markersize=10,
                        label=f"Simpleksų sk.: {len(traj)}",
                        markeredgecolor="black",
                        markeredgewidth=1.5,
                    )

        ax.set_xlabel("X₁", fontsize=12)
        ax.set_ylabel("X₂", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results, trajectories = run_optimization_comparison()
    visualize_optimization_paths(trajectories)
