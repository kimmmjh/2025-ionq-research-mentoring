import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from itertools import combinations


def weight(s: str):
    return sum([1 for char in s if char == "1"])


def alg_1(term_dict: dict, gamma: Parameter, num_qubits: int):
    """
    Algorithm 1 from the paper.

    This function first applies Rz gates for all single-qubit terms,
    and then greedily builds the CNOT network for multi-qubit terms, inserting
    their Rz gates as they are created.

    Args:
        term_dict (dict): A dictionary mapping bitstring parities to their weights.
        gamma (Parameter): The QAOA gamma parameter for the Rz rotation.
        num_qubits (int): The number of qubits.

    Returns:
        QuantumCircuit: The forward journey circuit with CNOTs and Rz gates.
    """
    if not term_dict:
        return QuantumCircuit(0)

    qc = QuantumCircuit(num_qubits)

    # Separate single-qubit terms from multi-qubit terms.
    single_qubit_terms = {s: w for s, w in term_dict.items() if weight(s) == 1}
    multi_qubit_terms = {s: w for s, w in term_dict.items() if weight(s) > 1}

    # Apply Rz gates for all single-qubit terms immediately.
    for s, w in single_qubit_terms.items():
        wire_index = s.find("1")
        angle = 2 * gamma * float(w)
        qc.rz(angle, wire_index)

    # Multi-qubit terms.
    S = {s: s for s in multi_qubit_terms.keys()}

    while S:
        s_min_orig, s_min_transformed = min(S.items(), key=lambda item: weight(item[1]))

        I = {i for i, char in enumerate(s_min_transformed) if char == "1"}
        control_qubit = min(I)
        target_qubit = min(I - {control_qubit})

        qc.cx(control_qubit, target_qubit)

        next_S = {}
        for s_orig, s_transformed in S.items():
            s_list = list(s_transformed)
            if s_list[control_qubit] == s_list[target_qubit]:
                s_list[control_qubit] = "0"
            else:
                s_list[control_qubit] = "1"
            s_new_transformed = "".join(s_list)

            if weight(s_new_transformed) == 1:
                wire_index = s_new_transformed.find("1")
                angle = 2 * gamma * float(multi_qubit_terms[s_orig])
                qc.rz(angle, wire_index)
            elif weight(s_new_transformed) > 1:
                next_S[s_orig] = s_new_transformed
            else:
                pass
        S = next_S
    return qc


def get_wire_matrix(qc: QuantumCircuit):
    """
    Function for determining the state of the parities for an input parity network (QuantumCircuit).
    """
    A = np.eye(qc.num_qubits, dtype=int)
    for instr in qc.data:
        if (
            instr.operation.name == "cx"
        ):  # Since there are rz gates now, we need to check for cx gates
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
    Return journey algotirhm.
    """
    n = A.shape[0]
    qc = QuantumCircuit(n)
    A_temp = A.copy()
    while A_temp.sum() > n:
        row_index_combinations = combinations(range(n), 2)
        l, m = max(
            row_index_combinations,
            key=lambda comb: bits_saved(A_temp[comb[0], :], A_temp[comb[1], :]),
        )

        Al = A_temp[l, :]
        Am = A_temp[m, :]
        if Al.sum() < Am.sum():
            qc.cx(l, m)
            A_temp[m, :] = (Al + Am) % 2
        else:
            qc.cx(m, l)
            A_temp[l, :] = (Al + Am) % 2
    return qc


def paulis_to_dict(hamiltonian_string, n_qubits):
    """
    Creates a dictionary mapping bitstrings to their weights from a Hamiltonian string.
    This version is updated to correctly handle negative weights and inconsistent spacing.
    """
    term_dict = {}

    hamil_string = hamiltonian_string.replace(" - ", " + -")

    terms = hamil_string.strip().split("+")

    for term in terms:
        term = term.strip()
        if not term:
            continue

        try:
            parts = term.split("*")
            if len(parts) != 2:
                raise ValueError("Term does not contain a single '*' separator.")

            weight_str = parts[0].strip()
            pauli_str = parts[1].strip()

            bitstring = ["0"] * n_qubits
            indices = "".join(filter(str.isdigit, pauli_str))
            for index in indices:
                bitstring[int(index)] = "1"

            term_dict["".join(bitstring)] = float(weight_str)
        except (ValueError, IndexError) as e:
            print(f"Skipping malformed term: '{term}'. Error: {e}")
            continue

    return term_dict


def main(n):
    gamma = Parameter("γ")
    strings = open("hamiltonian.txt").readlines()[0]

    term_dict = paulis_to_dict(strings, n)

    print("Parities and Weights of the Hamiltonian:")
    print(term_dict)

    qc_forward_with_rz = alg_1(term_dict, gamma, n)
    print("\nForward Circuit with Rz gates:")
    print(qc_forward_with_rz)

    wire_matrix = get_wire_matrix(qc_forward_with_rz)
    # print("\nEnding wire state of forward circuit:")
    # print(wire_matrix)

    inv = alg2(wire_matrix)
    print("\nReturn Journey Circuit:")
    print(inv)

    # Combine the forward and inverse circuits
    full_circuit = qc_forward_with_rz.compose(inv)
    print("\nFull Circuit:")
    print(full_circuit.decompose())

    # Verify that the final CNOT transformation is the identity
    print("\nFinal Wire State (should be identity):")
    final_matrix = get_wire_matrix(full_circuit.decompose())
    print(final_matrix)
    if np.allclose(final_matrix, np.eye(n)):
        print("Verification Successful!")
    else:
        print("Verification Failed.")

    return full_circuit, term_dict


if __name__ == "__main__":
    num_qubits = 8
    main(num_qubits)
