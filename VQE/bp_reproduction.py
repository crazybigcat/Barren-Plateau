from Library.QuantumSimulator import Simulator, Circuit, Gate
from Library.BasicFunSZZ import BasicFunctions_szz as bf
import torch as tc
import numpy as np


def random_pauli():
    tmp_rand = tc.rand(1)
    if tmp_rand < 1/3:
        return sigma_z
    elif tmp_rand > 2/3:
        return sigma_x
    else:
        return sigma_y


n_qubit = 18
dtype = tc.complex128
device = 'cpu'

# tc.autograd.set_detect_anomaly(True)
sigma_z = tc.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
sigma_x = tc.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
sigma_y = tc.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)


n_layer = 150

position_all = list(range(n_qubit))

a = Simulator.SimulatorProcess(n_qubit, device=device, dtype=dtype)

grad_list = []

para_matrix = tc.rand(n_qubit, n_layer, dtype=tc.float64, device=device, requires_grad=True)

ry_gate = Gate.Gate(tc.matrix_exp(1j*tc.pi*sigma_y/8))


for pp in range(n_qubit):
    a.circuit.add_single_gate(ry_gate, position=[pp])

for nn in range(n_layer):
    for pp in range(n_qubit):

        tmp_matrix = tc.linalg.matrix_exp(2j * tc.pi * para_matrix[pp, nn]*random_pauli())
        a.circuit.append(Gate.Gate(tmp_matrix, position=[pp], independent=False))
        # a.circuit.rand_gate(dim=2, position=[pp])

    for pp in range(n_qubit//2-1):
        a.circuit.phase_shift(position=[2*pp], control=[2*pp + 1])
        # a.circuit.not_gate(position=[2*pp], control=[2*pp + 1])
    for pp in range(n_qubit//2-1):
        a.circuit.phase_shift(position=[2*pp+1], control=[2*pp + 2])
        # a.circuit.not_gate(position=[2*pp+1], control=[2*pp + 2])

a.simulate(False)
# xx = a.fake_local_measure([n_qubit//2], operator=sigma_z)/a.state.norm()**2
xx = a.fake_local_measure([n_qubit//2], operator=sigma_z)
# xx = tc.abs(a.fake_local_measure([0], operator=sigma_z))
# xx = a.fake_local_measure([0], operator=sigma_z)/a.state.norm()**2
xx.backward()
print(xx.data)
big_num = 0


bf.easy_plot_3d(range(n_qubit), range(n_layer), para_matrix.grad.cpu().numpy())

print(para_matrix.grad.cpu().numpy())

np.savetxt('./tmp2.txt', para_matrix.grad.cpu().numpy())
