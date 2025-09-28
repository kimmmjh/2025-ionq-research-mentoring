import itertools

def verify_single_solution(clauses: list, bitstring: str) -> bool:
    for clause in clauses:
        is_clause_satisfied = False
        for qubit_index, required_value in clause:
            actual_value = int(bitstring[qubit_index])
            if actual_value == required_value:
                is_clause_satisfied = True
                break
        
        if not is_clause_satisfied:
            return False
            
    return True

def brute_force_k_sat_solver(num_qubits: int, clauses: list) -> list:
    solutions = []
    num_possible_solutions = 2**num_qubits
    
    print(f"Brute-forcing {num_possible_solutions} possible bitstrings for {num_qubits} variables...")

    for i in range(num_possible_solutions):
        bitstring = format(i, f'0{num_qubits}b')
        if verify_single_solution(clauses, bitstring):
            solutions.append(bitstring)
            
    print("Brute-force search complete.")
    return solutions


def verify_ksat_solution(clauses: list, bitstring: str) -> bool:
    for clause in clauses:
        is_clause_satisfied = False
        for qubit_index, required_value in clause:
            actual_value = int(bitstring[qubit_index])
            if actual_value == required_value:
                is_clause_satisfied = True
                break

        if not is_clause_satisfied:
            print(f"Verification FAILED. Bitstring '{bitstring}' violates clause: {clause}")
            return False

    print(f"Verification SUCCESS. Bitstring '{bitstring}' satisfies all clauses.")
    return True

num_qubits = 10
clauses = [
    [(0,0),(1,1),(2,1),(3,1),(4,0),(5,1),(6,1),(7,0),(8,1),(9,0)],
    [(0,1),(1,0),(2,0),(3,0),(4,0),(5,0),(6,1),(7,1),(8,0),(9,1)],
    [(0,0),(1,0),(2,1),(3,1),(4,1),(5,0),(6,1),(7,0),(8,1),(9,0)],
    [(0,1),(1,1),(2,0),(3,1),(4,1),(5,1),(6,0),(7,0),(8,0),(9,1)]
]

found_solutions = brute_force_k_sat_solver(num_qubits, clauses)

if not found_solutions:
    print("\nResult: No satisfying assignments were found.")
elif len(found_solutions) == 1:
    print(f"\nResult: Found a unique solution: {found_solutions[0]}")
else:
    print(f"\nResult: Found {len(found_solutions)} solutions:")
    for sol in found_solutions:
        print(f"  - {sol}")