from qiskit.quantum_info import SparsePauliOp
import re
from collections import defaultdict

num_qubits = 4
clauses = [
    [(0, 1), (1, 0), (2, 1)],  # x1 ∨ ¬x2 ∨ x3
    [(0, 0), (1, 1), (3, 1)],  # ¬x1 ∨ x2 ∨ x4
]

penalty = 1.0


def clause_to_pauli(clause, num_qubits):
    paulis = []
    coeffs = []
    terms = [(0, 1.0)]
    for qubit_index, val in clause:
        new_terms = []
        for bitmask, coeff in terms:
            if val == 1:
                new_terms.append((bitmask, coeff * 0.5))
                new_terms.append((bitmask | (1 << qubit_index), coeff * 0.5))
            else:
                new_terms.append((bitmask, coeff * 0.5))
                new_terms.append((bitmask | (1 << qubit_index), coeff * -0.5))
        terms = new_terms
    for bitmask, coeff in terms:
        pauli_str = "".join(
            ["Z" if (bitmask >> i) & 1 else "I" for i in range(num_qubits)]
        )
        paulis.append(pauli_str)
        coeffs.append(coeff)
    return SparsePauliOp(paulis, coeffs)


def format_hamiltonian_string(input_string: str) -> str:
    pattern = re.compile(r"([+-]?\d+\.\d+)\+\d+\.\d+j\s*\*\s*([IZ][IZ\d]*)")

    matches = pattern.findall(input_string)
    term_dict = defaultdict(float)

    for coeff_str, pauli_str in matches:
        coeff = float(coeff_str)
        term_dict[pauli_str] += coeff

    output_parts = []
    sorted_terms = sorted(term_dict.items(), key=lambda item: (len(item[0]), item[0]))

    for pauli_str, coeff in sorted_terms:
        if abs(coeff) < 1e-9:
            continue
        term_str = f"{coeff:.4f} * {pauli_str}"
        output_parts.append(term_str)

    final_string = " + ".join(output_parts)

    return final_string.replace("+ -", "- ")


hamiltonian = sum([clause_to_pauli(clause, num_qubits) * penalty for clause in clauses])

output_terms = []
for bitmask, coeff in zip(
    [p.to_label() for p in hamiltonian.paulis], hamiltonian.coeffs
):
    display_str = "".join([f"Z{i}" for i in range(num_qubits) if bitmask[i] == "Z"])
    if display_str == "":
        display_str = "I"
    coeff_str = f"{coeff:+.6f}"
    output_terms.append(f"{coeff_str} * {display_str}")


output_string = " ".join(output_terms)
print("Raw Hamiltonian string:\n", output_string)
output_string = format_hamiltonian_string(output_string)

print("The Hamiltonian is:\n", output_string)


with open("hamiltonian_output.txt", "w", encoding="utf-8") as f:
    f.write(output_string)
print("Exported to hamiltonian_output.txt")
