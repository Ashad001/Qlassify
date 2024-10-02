from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def ansatz_1(num_qubits, reps = 1):
    theta = ParameterVector('theta', length=num_qubits * 2)
    qc = QuantumCircuit(num_qubits)
    for layer in range(reps):
        for qubit in range(num_qubits):
            qc.ry(theta[layer * num_qubits + qubit], qubit)
        for qubit in range(num_qubits - 1):
            qc.cx(qubit, qubit + 1)
    return qc

def ansatz_2(num_qubits, reps = 1):
    theta = ParameterVector('theta', length=num_qubits * 4)
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

def ansatz_3(num_qubits, reps = 1):
    theta = ParameterVector('theta', length=num_qubits * 3)
    qc = QuantumCircuit(num_qubits)
    for layer in range(reps):
        for qubit in range(num_qubits):
            qc.rx(theta[layer * num_qubits + qubit], qubit)
            qc.ry(theta[layer * num_qubits + qubit + num_qubits], qubit)
            qc.rz(theta[layer * num_qubits + qubit + 2 * num_qubits], qubit)
        for qubit in range(num_qubits - 1):
            qc.cx(qubit, qubit + 1)
    
    return qc

def ansatz_4(num_qubits, reps = 1):
    theta = ParameterVector('theta', length=num_qubits * 5)
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
        # circular
        qc.cx(num_qubits - 1, 0)

    return qc

    