import numpy as np
from prettytable import PrettyTable

a, b, c = 8, 1, 7


def generate_simplex_matrix(a, b, c):
    return np.array(
        [
            [-1, 1, -1, -1, 1, 0, 0, 0, a],
            [2, 4, 0, 0, 0, 1, 0, 0, b],
            [0, 0, 1, 1, 0, 0, 1, 0, c],
            [2, -3, 0, -5, 0, 0, 0, 1, 0],
        ],
        dtype=float,
    )


def objective_function_value(X):
    """Skaičiuoja tikslo funkcijos reikšmę"""
    return 2 * X[0] - 3 * X[1] - 5 * X[3]


def simplex_algorithm(matrix, verbose=True):
    X_length = 4
    array_length = 9
    basis_length = 3
    basis = [4, 5, 6]
    inf = 1e10
    iteration = 0

    if verbose:
        print("PRADINĖ SIMPLEX LENTELĖ:")
        print_simplex_table(matrix, basis, iteration)

    index = np.argmin(matrix[-1][:-1])
    minimum = matrix[-1][index]

    while minimum < 0:
        iteration += 1

        ratios = []
        for i in range(basis_length):
            if matrix[i][index] <= 0:
                ratios.append(inf)
            else:
                ratio = matrix[i][-1] / matrix[i][index]
                ratios.append(ratio if ratio >= 0 else inf)

        pivot_index = np.argmin(ratios)

        if ratios[pivot_index] == inf:
            print("Uždavinys neturi sprendinio (neaprėžtas)!")
            return None, None, iteration

        if verbose:
            print(f"\nITERACIJA {iteration}:")
            print(f"Įeinantis kintamasis: x_{index + 1}")
            print(f"Išeinantis kintamasis: x_{basis[pivot_index] + 1}")

        matrix[pivot_index] = matrix[pivot_index] / matrix[pivot_index][index]

        for i in range(basis_length + 1):
            if i != pivot_index:
                multiplier = -matrix[i][index]
                for j in range(array_length):
                    matrix[i][j] = matrix[i][j] + multiplier * matrix[pivot_index][j]

        basis[pivot_index] = index

        if verbose:
            print_simplex_table(matrix, basis, iteration)

        index = np.argmin(matrix[-1][:-1])
        minimum = matrix[-1][index]

    return matrix, basis, iteration


def print_simplex_table(matrix, basis, iteration):
    """Atspausdina simplex lentelę"""
    table = PrettyTable()

    headers = ["Bazė"] + [f"x_{i + 1}" for i in range(7)] + ["z"] + ["Dešinė"]
    table.field_names = headers

    for i in range(3):
        row = [f"x_{basis[i] + 1}"] + [f"{matrix[i][j]:.4f}" for j in range(8)] + [f"{matrix[i][8]:.4f}"]
        table.add_row(row)

    z_row = ["z"] + [f"{matrix[3][j]:.4f}" for j in range(8)] + [f"{matrix[3][8]:.4f}"]
    table.add_row(z_row)

    print(table)


def extract_solution(matrix, basis):
    """Ištraukia sprendinį iš optimizuotos simplex lentelės"""
    X_length = 4
    X = [0.0] * X_length

    for i in range(X_length):
        if i in basis:
            X[i] = float(matrix[basis.index(i)][-1])

    return X


if __name__ == "__main__":
    matrix = generate_simplex_matrix(a, b, c)

    result_matrix, basis, iterations = simplex_algorithm(matrix.copy(), verbose=True)

    if result_matrix is None:
        exit()

    X_optimal = extract_solution(result_matrix, basis)
    f_min = objective_function_value(X_optimal)

    print("" + "=" * 100)
    print("SPRENDINIO REZULTATAI")
    print("=" * 100)

    results_table = PrettyTable()
    results_table.field_names = ["Parametras", "Reikšmė"]
    results_table.align["Parametras"] = "l"
    results_table.align["Reikšmė"] = "l"

    results_table.add_row(["Apribojimų konstantos (a, b, c)", f"({a}, {b}, {c})"])
    results_table.add_row(
        [
            "Optimalus sprendinys x*",
            f"x_1={X_optimal[0]:.6f}, x_2={X_optimal[1]:.6f}, x_3={X_optimal[2]:.6f}, x_4={X_optimal[3]:.6f}",
        ]
    )
    results_table.add_row(["Minimali tikslo funkcijos reikšmė", f"f(x*) = {f_min:.6f}"])
    results_table.add_row(["Baziniai kintamieji", f"x_{basis[0] + 1}, x_{basis[1] + 1}, x_{basis[2] + 1}"])
    results_table.add_row(["Iteracijų skaičius", f"{iterations}"])

    print(results_table)
