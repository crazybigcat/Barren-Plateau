from Library.QuantumSimulator import Simulator
import torch as tc
import numpy as np

n_qubit = 3
# ham = tc.eye(2**n_phi) * np.pi
a = Simulator.SimulatorProcess(n_qubit, device='cpu')
# a.sample(1000, list(range(n_phi, n_phi + n_qpe)))

tmp1 = a.state.clone()

a.circuit.qft([0, 1], inverse=True)
# print(a.state)
a.circuit.qft([0, 1], inverse=False)
a.simulate()

tmp2 = a.state.clone()
# a.sample(1000, list(range(n_phi, n_phi + n_qpe)))
print((tmp1 - tmp2).norm())
# print(tmp1)
# print(tmp2)
