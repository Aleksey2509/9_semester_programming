import math
import numpy as np
import matplotlib.pyplot as plt
import itertools
import functools
import scipy
import time
from tqdm import tqdm
import jax
import jax.numpy as jnp
import gc

def get_ccdm_general_table(k, n):
    table_size = 2 ** k
    table = [[0 for j in range(n)] for i in range(table_size)]

    for i in range(n):
        table[i + 1][i] = 1

    iota = [i for i in range(n)]

    disposition = n + 1
    for k in range(2, k):
        combinations = list(itertools.combinations(iota, k))
        len_combinations = len(combinations)

        for i, indices_tuple in enumerate(combinations):
            if i >= table_size - disposition:
                break
            table_ind = i + disposition
            for val in indices_tuple:
                table[table_ind][val] = 1
        
        disposition = disposition + len_combinations
    print(table)

    return table

def qam16_arr_create():
    qam16_arr = np.array([-3 + 3j, -3 + 1j, -3 - 3j, -3 - 1j,
                 -1 + 3j, -1 + 1j, -1 - 3j, -1 - 1j,
                  3 + 3j,  3 + 1j,  3 - 3j,  3 - 1j,
                  1 + 3j,  1 + 1j,  1 - 3j,  1 - 1j])
    # qam16_arr = qam16_arr * np.sqrt(len(qam16_arr)) / np.linalg.norm(qam16_arr)
    return qam16_arr, 4

def get_constellation_point_index(mapper_arr, point):
    diff = np.abs(mapper_arr - point)
    return np.argmin(diff)

def hard_slicer(points, mapper_arr, bits_depth):
    return np.array([mapper_arr[get_constellation_point_index(point, mapper_arr)] for point in points])

def demodulator(points, mapper_arr, bits_depth):
    indices = [get_constellation_point_index(point, mapper_arr) for point in points]
    bits_arr = []
    for i, ind in enumerate(indices):
        bits_arr.extend([int(digit) for digit in bin(ind)[2:].rjust(bits_depth, '0')])

    return np.array(bits_arr)

def get_bits(recv_points):
    mod_arr, bit_depth = qam16_arr_create()

    recv_bits = demodulator(recv_points.reshape(1, -1)[0], mod_arr, bit_depth)
    return recv_bits

def get_ser(ref_points, recv_points):
    ser = np.sum(np.logical_not(np.isclose(ref_points, recv_points)).astype(int)) / ref_points.size
    return ser

def get_ber(ref_bits, recv_bits):
    ber = np.sum(np.logical_xor(ref_bits, recv_bits)) / ref_bits.size
    return ber

def get_U_mat(input_symbols, M, offset):
    symb_size = input_symbols.shape[0] - 2 * offset
    X_x = np.array(input_symbols[:, 0])
    X_y = np.array(input_symbols[:, 1])
    piece_len = 2 * M + 1
    x_k_n = np.zeros((symb_size, piece_len * piece_len * 2), dtype=np.complex64)
    x_k_m = np.zeros((symb_size, piece_len * piece_len * 2), dtype=np.complex64)
    x_k_n_m = np.zeros((symb_size, piece_len * piece_len * 2), dtype=np.complex64)

    start = time.perf_counter()
    for k in tqdm(range(symb_size)):
        X_x_m_windowed = X_x[k + offset - M : k + offset + M + 1]
        X_y_m_windowed = X_y[k + offset - M : k + offset + M + 1]

        x_k_m[k, :piece_len * piece_len] = np.tile(X_x_m_windowed, (piece_len, 1)).T.reshape(1, -1)[0]
        x_k_m[k, piece_len * piece_len:] = x_k_m[k, :piece_len * piece_len]

        x_k_n[k, :piece_len * piece_len] = np.tile(X_x_m_windowed, (1, piece_len))[0]
        x_k_n[k, piece_len * piece_len:] = np.tile(X_y_m_windowed, (1, piece_len))[0]

        for m in range(-M, M + 1):
            start_ind_orig = k + offset + m
            start_ind_cur_vec_0 = (m + M) * piece_len
            start_ind_cur_vec_1 = (m + M) * piece_len + piece_len * piece_len

            x_k_n_m[k, start_ind_cur_vec_0 : start_ind_cur_vec_0 + piece_len] = X_x[start_ind_orig - M : start_ind_orig + M + 1]
            x_k_n_m[k, start_ind_cur_vec_1 : start_ind_cur_vec_1 + piece_len] = X_y[start_ind_orig - M : start_ind_orig + M + 1]

    x_k_n_m = x_k_n_m.conj()
    mid = time.perf_counter()

    U_x = np.multiply(np.multiply(x_k_m, x_k_n), x_k_n_m)
    # U_x = jnp.array(U_x)
    end = time.perf_counter()
    
    # print(f"took {end - mid} for mult and {mid - start} for init, res shape: {U_x.shape}")
    return U_x

def get_U_mat_simple(input_symbols, M, offset):
    symb_size = input_symbols.shape[0] - 2 * offset
    X_x = input_symbols[:, 0]
    X_y = input_symbols[:, 1]
    piece_len = 2 * M + 1
    U_x = np.zeros((symb_size, piece_len * piece_len * 2), dtype=np.complex64)

    for k in tqdm(range(symb_size)):
        for m in range(-M, M + 1):
            for n in range(-M, M + 1):
                m_disp = m + M
                n_disp = n + M
                U_x[k, m_disp * piece_len + n_disp] = X_x[k + m + offset] * X_x[k + n + offset] * X_x[k + m + n + offset].conj()
                U_x[k, piece_len * piece_len + m_disp * piece_len + n_disp] = X_x[k + m + offset] * X_y[k + n + offset] * X_y[k + m + n + offset].conj()
                # U_x = U_x.at[k, m_disp * piece_len + n_disp].set(X_x[k + m + offset] * X_x[k + n + offset] * X_x[k + m + n + offset].conj())
                # U_x = U_x.at[k, piece_len * piece_len + m_disp * piece_len + n_disp].set(X_x[k + m + offset] * X_y[k + n + offset] * X_y[k + m + n + offset].conj())
        # return U_x
        # jax.debug.print("Called")

    # print("Ended")
    return U_x

def get_U_mat_from_tensor(input_symbols, M, offset):
    symb_size = input_symbols.shape[0] - 2 * offset
    X_x = np.array(input_symbols[:, 0])
    X_y = np.array(input_symbols[:, 1])
    piece_len = 2 * M + 1
    U_x = np.zeros((symb_size, 2, piece_len, piece_len), dtype=np.complex64)

    for k in tqdm(range(symb_size)):
        for m in range(-M, M + 1):
            for n in range(-M, M + 1):
                m_disp = m + M
                n_disp = n + M
                U_x[k, 0, m_disp, n_disp] = X_x[k + m + offset] * X_x[k + n + offset] * np.conj(X_x[k + m + n + offset])
                U_x[k, 1, m_disp, n_disp] = X_x[k + m + offset] * X_y[k + n + offset] * np.conj(X_y[k + m + n + offset])
    U_x = U_x.reshape(symb_size, -1)
    return U_x

def get_nmse_and_ber(matrices_dict, M, offset):

    model_input = matrices_dict['eqSymOutData']
    model_input_cut = model_input[offset : -offset]
    model_expected = matrices_dict['srcSymData'][offset : -offset]
    model_expected_bits = matrices_dict['srcPermBitData'][offset * 4 : -4 * offset]
    # breakpoint()
    denom = np.sum(np.square(np.abs(model_expected)))

    if M == 0: # without model
        model_output = model_input[offset : -offset]
        nmse = np.sum(np.square(np.abs(model_output - model_expected))) / denom

        recv_points = hard_slicer(model_output.reshape(1, -1)[0], *qam16_arr_create())
        ser = get_ser(model_expected.reshape(1, -1)[0], recv_points)

        recv_bits = get_bits(recv_points)
        ref_bits = get_bits(model_expected.reshape(1, -1)[0])
        ber = get_ber(ref_bits, recv_bits)
        return nmse, ser, ber

    model_swapped = np.array(np.concat((model_input[:, [1]], model_input[:, [0]]), axis=1))
    # get_U_simple_jaxed = jax.jit(get_U_mat_simple, static_argnames=("M", "offset"))(PBM_input, M, offset)
    # print("compiled")
    symb_amount = model_input.shape[0] - 2 * offset
    U = np.zeros((2, symb_amount, 2 * (2 * M + 1) * (2 * M + 1)), dtype=np.complex64)
    U[0] = get_U_mat(model_input, M, offset)
    gc.collect()
    U[1] = get_U_mat(model_swapped, M, offset)
    gc.collect()
    print("ended creating U matrix")

    # d_vec = jnp.array(d_vec)
    c = np.matmul(np.linalg.pinv(U), np.expand_dims((model_expected - model_input_cut).T, 2))
    gc.collect()
    model_output = np.squeeze(np.matmul(U, c), 2).T + model_input_cut
    gc.collect()

    nmse = np.sum(np.square(np.abs(model_output - model_expected))) / denom
    gc.collect()

    recv_points = hard_slicer(model_output.reshape(1, -1)[0], *qam16_arr_create())
    ser = get_ser(model_expected.reshape(1, -1)[0], recv_points)

    ref_bits = get_bits(model_expected.reshape(1, -1)[0])
    recv_bits = get_bits(recv_points)
    ber = get_ber(ref_bits, recv_bits)

    gc.collect()
    return nmse, ser, ber

def whole_experiment():
    offset = 100000 # good value
    # offset = 70000
    # offset = 100
    res = scipy.io.loadmat("/home/lexotr/my_gits/9_semester_programming/optics/second_hw/pbm_test.mat")
    matrices_dict = {}
    for key in res:
        if str(key).startswith("__"):
            continue
        matrices_dict[key] = res[key]
    no_model_nmse, no_model_ser, no_model_ber = get_nmse_and_ber(matrices_dict, 0, offset)
    print(f"No model: NMSE: {no_model_nmse}, NMSE(dB): {10 * np.log10(no_model_nmse)}, SER: {no_model_ser}, BER: {no_model_ber}")
    M_range = [5, 10]
    # M_range = [5]
    for M in M_range:
        model_nmse, model_ser, model_ber = get_nmse_and_ber(matrices_dict, M, offset)
        print(f"M = {M}: NMSE: {model_nmse}, NMSE(dB): {10 * np.log10(model_nmse)}, SER: {model_ser}, BER: {model_ber}")
        gc.collect()

if __name__ == "__main__":
    whole_experiment()
