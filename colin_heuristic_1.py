# This is from Colin Campbell - February 20, 2025

import numpy as np
from qiskit import QuantumCircuit
from itertools import combinations


def weight(s: str):
    return sum([1 for char in s if char == "1"])


def alg_1(S: set[str]):
    """
    Algorithm 1 in the paper, designed to greedily build the parity netweork for an input set of bit strings
    S: set of bit strings representing products of pauli Z operators
    return: QuantumCircuit representing a parity network hitting each bit string in S at least once
    """
    n = len(next(iter(S)))
    qc = QuantumCircuit(n)
    S = {s for s in S if weight(s) != 1}  # First order terms are "free"
    while S:
        s_min = min(S, key=weight)
        I = set(i for i, char in enumerate(s_min) if char == "1")
        i = min(I)
        j = min(I - {i})
        qc.cx(i, j)
        S_prime = set()
        for s in S:
            s_prime = s
            new_bit = str((int(s_prime[i]) + int(s_prime[j])) % 2)
            s_prime = "".join(
                [char if idx != i else new_bit for idx, char in enumerate(s_prime)]
            )
            if weight(s_prime) > 1:
                S_prime.add(s_prime)
        S = S_prime
    return qc


def get_wire_matrix(qc: QuantumCircuit):
    """
    Function for determining the state of the parities for an input parity network (QuantumCircuit). Nonzero values in a row represent the parities included in the wire of that row. For example, a row such as [1, 0, 0, 1, 0] represents a wire having parity x_0 \oplus x_3.
    qc: QuantumCircuit representing a parity network (only includes CX gates)
    return: boolean array representing the current state of the wires after the parity network.
    """
    A = np.eye(qc.num_qubits)
    for instr in qc.data:
        if instr.operation.name != "cx":
            raise ValueError("Circuit is not a parity network")
        control, target = [q._index for q in instr.qubits]
        A[target, :] = (A[control, :] + A[target, :]) % 2
    return A


def bits_saved(row_i, row_j):
    """
    Calculates the number of entries cancelled when adding two rows together (mod 2). Cancelling more entries is favorable because it means fewer CX gates in the remaining circuit.
    row_i: numpy array of boolean values
    row_j: numpy array of boolean values
    return: integer representing the amount of cancellation achieved by adding the rows together (mod 2)
    """
    big_row = max(row_i, row_j, key=sum)
    row_sum = (row_i + row_j) % 2
    return_value = big_row.sum() - row_sum.sum()
    return return_value


def alg2(A: np.array):
    """
    Algorithm 2 from the paper. Performd greedy elimination to produce a circuit that can be used to invert a parity network so that it points at the identity (or permutation therof).
    A: boolean array representing the wire after a parity network
    return: QuantumCircuit inverting the wire state based on a greedy elimination heuristic
    """
    n = A.shape[0]
    qc = QuantumCircuit(n)
    while A.sum() > n:
        row_index_combinations = combinations(range(n), 2)
        l, m = max(
            row_index_combinations,
            key=lambda comb: bits_saved(A[comb[0], :], A[comb[1], :]),
        )
        Al = A[l, :]
        Am = A[m, :]
        if Al.sum() < Am.sum():
            qc.cx(l, m)
            A[m, :] = (Al + Am) % 2
        else:
            qc.cx(m, l)
            A[l, :] = (Al + Am) % 2
    return qc


def string_to_paulis(strings):
    weights = []
    paulis = []
    terms = strings.split(" + ")

    for term in terms:
        weight, pauli = term.split("*")

        weights.append(weight)
        paulis.append(pauli)

    return weights, paulis


def paulis_to_bitstrings(paulis, n_qubits):
    S = set()
    pos = []
    for pauli in paulis:
        temp = []
        for s in pauli:
            if s.isdigit():
                temp.append(s)
            else:
                continue
        pos.append(temp)
    for p in pos:
        bitstring = ["0"] * n_qubits
        for index in p:
            bitstring[int(index)] = "1"
        S.add("".join(bitstring))

    return S


if __name__ == "__main__":
    strings = open("hamiltonian.txt").readlines()[0]
    weights, paulis = string_to_paulis(strings)
    num_qubits = 5
    S_example = paulis_to_bitstrings(paulis, num_qubits)
    print("Starting Bit Strings:")
    print(S_example)

    qc_example = alg_1(S_example)
    print("Parity Network:")
    print(qc_example)

    wire_matrix = get_wire_matrix(qc_example)
    print("Ending wire state:")
    print(wire_matrix)

    inv = alg2(wire_matrix)
    print("Inverting Circuit:")
    print(inv)
    qc_example.append(inv, range(qc_example.num_qubits))
    print("Full Circuit:")
    print(qc_example.decompose())
    print("Final Wire State:")
    print(get_wire_matrix(qc_example.decompose()))
