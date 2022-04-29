import scipy.io as io
import torch as tc
import numpy as np
import math


def safe_svd(matrix):
    if tc.any(tc.isnan(matrix)):
        print('!!!!!!!!!!')
        print(matrix)
    try:
        u, s, v = tc.linalg.svd(matrix, full_matrices=False)
    except RuntimeError:
        device = matrix.device
        dtype = matrix.dtype
        u, s, v = np.linalg.svd(matrix.cpu())
        u = tc.from_numpy(u).to(device).to(dtype)
        s = tc.from_numpy(s).to(device).to(dtype)
        v = tc.from_numpy(v).to(device).to(dtype)
    if tc.any(tc.isnan(u)) or tc.any(tc.isnan(v)) or tc.any(tc.isnan(s)):
        device = matrix.device
        dtype = matrix.dtype
        u, s, v = np.linalg.svd(matrix.cpu())
        u = tc.from_numpy(u).to(device).to(dtype)
        s = tc.from_numpy(s).to(device).to(dtype)
        v = tc.from_numpy(v).to(device).to(dtype)

    return u, s, v

def load_mat(file_name, file_path='./'):
    merged_name = file_path + file_name
    data_loaded = io.loadmat(merged_name)
    return data_loaded

def load_mat_fdz_20211118(file_name, file_path='./', device='cuda:0'):
    merged_name = file_path + file_name
    data_loaded = io.loadmat(merged_name)
    T_name_list = ['X', 'Y', 'Z', 'A', 'B', 'C']
    lm_name_list = ['XA', 'XB', 'XC', 'YA', 'YB', 'YC', 'ZA', 'ZB', 'ZC']
    T = dict()
    lm = dict()
    for nn in range(len(T_name_list)):
        T[T_name_list[nn]] = tc.from_numpy(data_loaded['T'][0,0][nn]).to(device)
    for nn in range(len(lm_name_list)):
        lm[lm_name_list[nn]] = tc.from_numpy(data_loaded['lm'][0,0][nn]).to(device)
        lm[lm_name_list[nn]] = tc.diag(lm[lm_name_list[nn]])
    new_T_list = ['X', 'A', 'Y', 'B', 'Z', 'C']
    new_lm_list = ['XA', 'YA', 'YB', 'ZB', 'ZC', 'XC', 'XB', 'ZA', 'YC', 'XB', 'ZA', 'YC']
    new_T = list()
    new_lm = list()
    for name in new_T_list:
        new_T.append(T[name])
    for name in new_lm_list:
        new_lm.append(lm[name])
    return new_T, new_lm

def load_mat_fdz_20220124(file_name, file_path='./', device='cuda:0'):
    merged_name = file_path + file_name
    data_loaded = io.loadmat(merged_name)
    T_name_list = ['X', 'Y', 'Z', 'A', 'B', 'C']
    lm_name_list = ['XA', 'XB', 'XC', 'YA', 'YB', 'YC', 'ZA', 'ZB', 'ZC']
    T = dict()
    lm = dict()
    for nn in range(len(T_name_list)):
        T[T_name_list[nn]] = tc.from_numpy(data_loaded['T'][0,0][nn]).to(device)
    for nn in range(len(lm_name_list)):
        lm[lm_name_list[nn]] = tc.from_numpy(data_loaded['lm'][0,0][nn]).to(device)
        lm[lm_name_list[nn]] = tc.diag(lm[lm_name_list[nn]])
    new_T_list = ['X', 'A', 'Y', 'B', 'Z', 'C']
    new_lm_list = ['XA', 'XB', 'XC', 'YA', 'YB', 'YC', 'ZA', 'ZB', 'ZC']
    new_T = list()
    new_lm = list()
    for name in new_T_list:
        new_T.append(T[name])
    for name in new_lm_list:
        new_lm.append(lm[name])
    return new_T, new_lm



def np_kron(*args):
    tmp = args[0]
    for nn in range(1, len(args)):
        tmp = np.kron(tmp, args[nn])
    return tmp


def outer_parallel(a, *matrix):
    # need optimization
    for b in matrix:
        a = (a.repeat(b.shape[1], 1).reshape(a.shape + (-1,))
             * b.repeat(a.shape[1], 0).reshape(a.shape + (-1,))).reshape(a.shape[0], -1)
    return a


def outer(a, *matrix):
    for b in matrix:
        a = np.outer(a, b).flatten()
    return a


def tensor_contract(a, b, index):
    ndim_a = np.array(a.shape)
    ndim_b = np.array(b.shape)
    order_a = np.arange(len(ndim_a))
    order_b = np.arange(len(ndim_b))
    order_a_contract = np.array(order_a[index[0]]).flatten()
    order_b_contract = np.array(order_b[index[1]]).flatten()
    order_a_hold = np.setdiff1d(order_a, order_a_contract)
    order_b_hold = np.setdiff1d(order_b, order_b_contract)
    hold_shape_a = ndim_a[order_a_hold].flatten()
    hold_shape_b = ndim_b[order_b_hold].flatten()
    return np.dot(
        a.transpose(np.concatenate([order_a_hold, order_a_contract])).reshape(hold_shape_a.prod(), -1),
        b.transpose(np.concatenate([order_b_contract, order_b_hold])).reshape(-1, hold_shape_b.prod())) \
        .reshape(np.concatenate([hold_shape_a, hold_shape_b]))


def tensor_svd(tmp_tensor, index_left='none', index_right='none'):
    tmp_shape = np.array(tmp_tensor.shape)
    tmp_index = np.arange(len(tmp_tensor.shape))
    if index_left == 'none':
        index_right = tmp_index[index_right].flatten()
        index_left = np.setdiff1d(tmp_index, index_right)
    elif index_right == 'none':
        index_left = tmp_index[index_left].flatten()
        index_right = np.setdiff1d(tmp_index, index_left)
    index_right = np.array(index_right).flatten()
    index_left = np.array(index_left).flatten()
    tmp_tensor = tmp_tensor.transpose(np.concatenate([index_left, index_right]))
    tmp_tensor = tmp_tensor.reshape(tmp_shape[index_left].prod(), tmp_shape[index_right].prod())
    u, l, v = np.linalg.svd(tmp_tensor, full_matrices=False)
    return u, l, v


def calculate_mse(image_origin, image_noised):
    return ((image_origin - image_noised) ** 2) / np.prod(image_origin.shape)


def calculate_psnr(image_origin, image_noised):
    # not very correct
    image_origin = image_origin.flatten()
    image_noised = image_noised.flatten()
    return 20 * np.log10(
        np.concatenate((image_origin, image_noised)).max()) - 10 * np.log10(
        ((image_origin - image_noised) ** 2).sum() / np.prod(image_origin.shape))


def noise_image(image_origin, noise_type='Gaussian', means=0, sigma=1):
    pixel_max = image_origin.max()
    pixel_min = image_origin.min()

# QHES related

def get_x_qpe(n):
    rx = -(n + 2)/math.log2(1 - (2/math.pi)**2)
    x = int(rx) + 2
    return x


def get_m_qpe(n):
    x = get_x_qpe(n)
    m = 0
    result_p = 0
    while result_p < 0.8:
        m = m + 1
        tmp_x = math.pi/(2 * m)
        result_p = (math.sin(tmp_x)/tmp_x) ** (2*x)
    return m

def np_kron(*args):
    tmp = args[0]
    for nn in range(1, len(args)):
        tmp = np.kron(tmp, args[nn])
    return tmp

# QHES end

if __name__ == '__main__':
    print('This is a pure function file. What are you doing?')

