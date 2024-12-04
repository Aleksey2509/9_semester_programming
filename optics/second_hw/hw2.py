import math
import numpy as np
import matplotlib.pyplot as plt
import itertools
import functools
import scipy
import time
from tqdm import tqdm
import gc

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

def get_U_mat(input_symbols, M, offset, symb_size):
    X_x = np.array(input_symbols[:, 0])
    X_y = np.array(input_symbols[:, 1])
    piece_len = 2 * M + 1
    x_k_n = np.zeros((symb_size, piece_len * piece_len * 2), dtype=np.complex128)
    x_k_m = np.zeros((symb_size, piece_len * piece_len * 2), dtype=np.complex128)
    x_k_n_m = np.zeros((symb_size, piece_len * piece_len * 2), dtype=np.complex128)

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
    U_x += 1e-10 * np.eye(U_x.shape[0], U_x.shape[1])
    # U_x = jnp.array(U_x)
    end = time.perf_counter()
    
    # print(f"took {end - mid} for mult and {mid - start} for init, res shape: {U_x.shape}")
    return U_x


def get_model_output(model_input, model_input_cut, model_expected, M, offset, symb_size):
    if M == 0: # without model
        return model_input_cut
    model_swapped = np.array(np.concat((model_input[:, [1]], model_input[:, [0]]), axis=1))
    U = np.zeros((2, symb_size, 2 * (2 * M + 1) * (2 * M + 1)), dtype=np.complex128)
    U[0] = get_U_mat(model_input, M, offset, symb_size)
    gc.collect()
    U[1] = get_U_mat(model_swapped, M, offset, symb_size)
    gc.collect()
    print("ended creating U matrix")

    # d_vec = jnp.array(d_vec)
    U_pinv = np.linalg.pinv(U)
    print("Evaluated pinv")
    gc.collect()
    c = np.matmul(U_pinv, np.expand_dims((model_expected - model_input_cut).T, 2))
    print("Evaluated the coefficients")
    gc.collect()
    model_output = np.squeeze(np.matmul(U, c), 2).T + model_input_cut
    print("Got the model output")
    gc.collect()
    return model_output


def get_nmse_and_ber(matrices_dict, M, offset, symb_size):

    model_input = matrices_dict['eqSymOutData']
    model_input_cut = model_input[offset : symb_size + offset]
    model_expected = matrices_dict['srcSymData'][offset : symb_size + offset]

    model_output = get_model_output(model_input, model_input_cut, model_expected, M, offset, symb_size)
    gc.collect()
    # breakpoint()
    denom = np.sum(np.square(np.abs(model_expected)))
    denom_x = np.sum(np.square(np.abs(model_expected[:, 0])))
    denom_y = np.sum(np.square(np.abs(model_expected[:, 1])))

    if M == 0: # without model
        nmse = np.sum(np.square(np.abs(model_output - model_expected))) / denom
        nmse_x = np.sum(np.square(np.abs(model_output[:, 0] - model_expected[:, 0]))) / denom_x
        nmse_y = np.sum(np.square(np.abs(model_output[:, 1] - model_expected[:, 1]))) / denom_y

        ber_x_y = [0,0]
        ser_x_y = [0,0]
        for i in range(2):
            recv_points = hard_slicer(model_output[:, i], *qam16_arr_create())
            ser_x_y[i] = get_ser(model_expected[:, i], recv_points)
            
            recv_bits = get_bits(recv_points)
            ref_bits = get_bits(model_expected[:, i])
            ber_x_y[i] = get_ber(ref_bits, recv_bits)
        ber_x, ber_y = ber_x_y
        ser_x, ser_y = ser_x_y

        recv_points = hard_slicer(model_output.reshape(1, -1)[0], *qam16_arr_create())
        ser = get_ser(model_expected.reshape(1, -1)[0], recv_points)

        recv_bits = get_bits(recv_points)
        ref_bits = get_bits(model_expected.reshape(1, -1)[0])
        ber = get_ber(ref_bits, recv_bits)
        return (nmse_x, nmse_y, nmse), (ser_x, ser_y, ser), (ber_x, ber_y, ber)

    nmse = np.sum(np.square(np.abs(model_output - model_expected))) / denom
    nmse_x = np.sum(np.square(np.abs(model_output[:, 0] - model_expected[:, 0]))) / denom_x
    nmse_y = np.sum(np.square(np.abs(model_output[:, 1] - model_expected[:, 1]))) / denom_y
    gc.collect()
    
    ber_x_y = [0,0]
    ser_x_y = [0,0]
    for i in range(2):
        recv_points = hard_slicer(model_output[:, i], *qam16_arr_create())
        ser_x_y[i] = get_ser(model_expected[:, i], recv_points)
        
        recv_bits = get_bits(recv_points)
        ref_bits = get_bits(model_expected[:, i])
        ber_x_y[i] = get_ber(ref_bits, recv_bits)
    ber_x, ber_y = ber_x_y
    ser_x, ser_y = ser_x_y

    recv_points = hard_slicer(model_output.reshape(1, -1)[0], *qam16_arr_create())
    ser = get_ser(model_expected.reshape(1, -1)[0], recv_points)

    ref_bits = get_bits(model_expected.reshape(1, -1)[0])
    recv_bits = get_bits(recv_points)
    ber = get_ber(ref_bits, recv_bits)

    gc.collect()
    return (nmse_x, nmse_y, nmse), (ser_x, ser_y, ser), (ber_x, ber_y, ber)

def whole_experiment():
    # offset = 100000 # good value
    offset = 100000
    symb_size = 50000
    # offset = 50000
    # offset = 200
    res = scipy.io.loadmat("./pbm_test.mat")
    matrices_dict = {}
    for key in res:
        if str(key).startswith("__"):
            continue
        matrices_dict[key] = res[key]
    no_model_nmse_tuple, no_model_ser_tuple, no_model_ber_tuple = get_nmse_and_ber(matrices_dict, 0, offset, symb_size)
    nmse_arr = []
    ber_arr = []
    nmse_x_arr = []
    ber_x_arr = []
    nmse_y_arr = []
    ber_y_arr = []
    print(f"No model: NMSE: {no_model_nmse_tuple[2]}, NMSE(dB): {10 * np.log10(no_model_nmse_tuple[2])}, SER: {no_model_ser_tuple[2]}, BER: {no_model_ber_tuple[2]}")
    M_range = range(0, 21)
    # M_range = range(11)
    # M_range = [11]
    for M in M_range:
        model_nmse_tuple, model_ser_tuple, model_ber_tuple = get_nmse_and_ber(matrices_dict, M, offset, symb_size)
        nmse_x_arr.append(model_nmse_tuple[0])
        ber_x_arr.append(model_ber_tuple[0])
        nmse_y_arr.append(model_nmse_tuple[1])
        ber_y_arr.append(model_ber_tuple[1])
        nmse_arr.append(model_nmse_tuple[2])
        ber_arr.append(model_ber_tuple[2])
        print(f"M = {M}: NMSE: {model_nmse_tuple[2]}, NMSE(dB): {10 * np.log10(model_nmse_tuple[2])}, SER: {model_ser_tuple[2]}, BER: {model_ber_tuple[2]}")
        gc.collect()
        np.save(f"./npy_saves/nmsex_{offset}_{symb_size}.npy", np.array(nmse_x_arr))
        np.save(f"./npy_saves/berx_{offset}_{symb_size}.npy", np.array(ber_x_arr))
        np.save(f"./npy_saves/nmsey_{offset}_{symb_size}.npy", np.array(nmse_y_arr))
        np.save(f"./npy_saves/bery_{offset}_{symb_size}.npy", np.array(ber_y_arr))
        np.save(f"./npy_saves/nmse_{offset}_{symb_size}.npy", np.array(nmse_arr))
        np.save(f"./npy_saves/ber_{offset}_{symb_size}.npy", np.array(ber_arr))
        gc.collect()
    plt.plot(M_range, 10 * np.log10(np.array(nmse_arr)))
    plt.grid()
    plt.xlabel('M')
    plt.ylabel('NMSE')
    plt.title('NMSE(M)')
    plt.savefig("NMSE_M.png")
    plt.show()
    
    plt.plot(M_range, ber_arr)
    plt.yscale('log')
    plt.grid()
    plt.xlabel('M')
    plt.ylabel('BER')
    plt.title('BER(M)')
    plt.savefig("BER_M.png")
    plt.show()

if __name__ == "__main__":
    print('This script expects dir "./npy_saves" to exist to save results to it') 
    whole_experiment()
