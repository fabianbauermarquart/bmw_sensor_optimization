from docplex.mp.model import Model
from qiskit import BasicAer
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.algorithms import VQE
from qiskit.optimization import QuadraticProgram
from qiskit.optimization.algorithms import MinimumEigenOptimizer

from utils.types import flatten_list


def main():
    # Problem
    list_of_subsets = [[1], [1, 2, 3], [3, 4], [5]]

    n = len(set(flatten_list(list_of_subsets)))
    N = len(list_of_subsets)

    A = 1.0

    # build model with docplex
    mdl = Model()
    x = [mdl.binary_var() for _ in range(N)]

    objective = A * mdl.sum((1 - mdl.sum(x[i] for i in range(N)
                                         if alpha in list_of_subsets[i])) ** 2
                            for alpha in range(n))
    mdl.minimize(objective)

    # convert to Qiskit's quadratic program
    qp = QuadraticProgram()
    qp.from_docplex(mdl)

    print(qp)

    aqua_globals.random_seed = 10598
    quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                       seed_simulator=aqua_globals.random_seed,
                                       seed_transpiler=aqua_globals.random_seed)

    vqe = VQE(quantum_instance=quantum_instance)
    optimizer = MinimumEigenOptimizer(min_eigen_solver=vqe)
    result = optimizer.solve(qp)

    print(result)


if __name__ == "__main__":
    main()
