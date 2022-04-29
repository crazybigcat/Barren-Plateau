from Library.QuantumSimulator import Simulator as qs
import torch as tc

# device
device = 'cpu'
dtype = tc.complex64

# number of qubits

n_qubit = 4

b = qs.SimulatorProcess(n_qubit, device=device, dtype=dtype)

b.sampling(1000, basis='xxxx')
print('!!!!!!')
b.circuit.hadamard(list(range(n_qubit)))
b.simulate()
b.sampling(1000)
b.sampling(1000, basis='xxxx')

