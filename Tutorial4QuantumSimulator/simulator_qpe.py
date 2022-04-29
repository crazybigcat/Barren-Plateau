from Library.QuantumSimulator import Simulator
from Library.QuantumSimulator import Gate
from Library.QuantumSimulator import Circuit
from Library.BasicFunSZZ import physical_fun as pf
from scipy.linalg import eigh
import torch as tc
import numpy as np

# number of qubits

n_phi = 3
n_qpe = 4
n_qubit = n_phi + n_qpe

# device and dtype
device = 'cpu'
dtype = tc.complex64

# initialize position

position_phi = list(range(n_phi))
position_qpe = list(range(n_phi, n_phi + n_qpe))

# definite

ham = pf.generate_hamiltonian(n_phi)
ham = ham / (n_phi - 1)

ham = ham + 3*tc.eye(2 ** n_phi).numpy()

eigv, eigs = eigh(ham, eigvals=(0, 0))
print(eigv)
# print(eigs)
# ham = ham * 0.75 * 2 * np.pi
ham = tc.eye(2**n_phi).numpy() / 2
ham = ham * 2 * np.pi * 0.75
ham = tc.from_numpy(ham)
# ham = ham.to_sparse()


b = Simulator.SimulatorProcess(n_qubit, device=device, dtype=dtype)
unitary = Gate.time_evolution(ham, 1).to(device).to(dtype)
tmp_circuit = Circuit.Circuit(n_phi)
tmp_circuit.compose(unitary, position=list(range(n_phi)))
# b.circuit.qpe(unitary, position_phi, position_qpe)
b.circuit.qpe(tmp_circuit, position_phi, position_qpe)
# b.circuit.qpe(unitary,position_phi, position_qpe, inverse=True)
b.simulate()
b.sampling(1024, position=position_qpe)