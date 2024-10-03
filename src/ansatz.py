from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def ansatz_1(num_qubits, reps=1):
    # Only requires 1 param per qubit per layer
    theta = ParameterVector('theta', length=num_qubits * reps)
    qc = QuantumCircuit(num_qubits)
    for layer in range(reps):
        for qubit in range(num_qubits):
            qc.ry(theta[layer * num_qubits + qubit], qubit)
        for qubit in range(num_qubits - 1):
            qc.cx(qubit, qubit + 1)
    return qc

def ansatz_2(num_qubits, reps=1):
    # Requires 3 params per qubit per layer (rx, ry, and rz)
    theta = ParameterVector('theta', length=num_qubits * reps * 3)
    qc = QuantumCircuit(num_qubits)
    for layer in range(reps):
        for qubit in range(num_qubits):
            qc.rx(theta[layer * num_qubits + qubit], qubit)
            qc.ry(theta[layer * num_qubits + qubit + num_qubits], qubit)
        for qubit in range(num_qubits - 1):
            qc.cx(qubit, qubit + 1)
        for qubit in range(num_qubits):
            qc.rz(theta[layer * num_qubits + qubit + 2 * num_qubits], qubit)
    return qc

def ansatz_3(num_qubits, reps=1):
    # Also requires 3 params per qubit per layer (rx, ry, and rz)
    theta = ParameterVector('theta', length=num_qubits * reps * 3)
    qc = QuantumCircuit(num_qubits)
    for layer in range(reps):
        for qubit in range(num_qubits):
            qc.rx(theta[layer * num_qubits + qubit], qubit)
            qc.ry(theta[layer * num_qubits + qubit + num_qubits], qubit)
            qc.rz(theta[layer * num_qubits + qubit + 2 * num_qubits], qubit)
        for qubit in range(num_qubits - 1):
            qc.cx(qubit, qubit + 1)
    return qc

def ansatz_4(num_qubits, reps=1):
    # Requires 5 params per qubit per layer (rx, ry, rz, rx, ry)
    theta = ParameterVector('theta', length=num_qubits * reps * 5)
    qc = QuantumCircuit(num_qubits)
    for layer in range(reps):
        for qubit in range(num_qubits):
            qc.rx(theta[layer * num_qubits + qubit], qubit)
            qc.ry(theta[layer * num_qubits + qubit + num_qubits], qubit)
            qc.rz(theta[layer * num_qubits + qubit + 2 * num_qubits], qubit)
            qc.rx(theta[layer * num_qubits + qubit + 3 * num_qubits], qubit)
            qc.ry(theta[layer * num_qubits + qubit + 4 * num_qubits], qubit)
        for qubit in range(num_qubits - 1):
            qc.cx(qubit, qubit + 1)
        # for Circular connection
        qc.cx(num_qubits - 1, 0)
    return qc
