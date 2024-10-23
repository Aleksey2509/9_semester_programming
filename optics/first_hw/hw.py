import math
import numpy as np
import matplotlib.pyplot as plt

def qpsk_arr_create():
    qpsk_arr = np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j])
    qpsk_arr = qpsk_arr * np.sqrt(len(qpsk_arr)) / np.linalg.norm(qpsk_arr)
    return qpsk_arr, 2

def qam16_arr_create():
    qam16_arr = np.array([-3 + 3j, -3 + 1j, -3 - 3j, -3 - 1j,
                 -1 + 3j, -1 + 1j, -1 - 3j, -1 - 1j,
                  3 + 3j,  3 + 1j,  3 - 3j,  3 - 1j,
                  1 + 3j,  1 + 1j,  1 - 3j,  1 - 1j])
    qam16_arr = qam16_arr * np.sqrt(len(qam16_arr)) / np.linalg.norm(qam16_arr)
    return qam16_arr, 4

def slower_qam64(bits):
    ampl_arr = [7, 5, 1, 3]
    
    bits = int(bits)
    lower_3_bits = bits % 8

    y_sign = -1
    if lower_3_bits >= 4:
        y_sign = +1

    ampl_y = ampl_arr[lower_3_bits % 4] * y_sign

    higher_3_bits = bits // 8
    x_sign = -1
    if higher_3_bits >= 4:
        x_sign = +1

    ampl_x = ampl_arr[higher_3_bits % 4] * x_sign

    return ampl_x + 1j * ampl_y



def qam64_arr_create():
    qam64_arr = np.array([-7 - 7j, -7 - 5j, -7 - 1j, -7 - 3j,
                          -7 + 7j, -7 + 5j, -7 + 1j, -7 + 3j,
                          -5 - 7j, -5 - 5j, -5 - 1j, -5 - 3j,
                          -5 + 7j, -5 + 5j, -5 + 1j, -5 + 3j,
                          -1 - 7j, -1 - 5j, -1 - 1j, -1 - 3j,
                          -1 + 7j, -1 + 5j, -1 + 1j, -1 + 3j,
                          -3 - 7j, -3 - 5j, -3 - 1j, -3 - 3j,
                          -3 + 7j, -3 + 5j, -3 + 1j, -3 + 3j,
                          +7 - 7j, +7 - 5j, +7 - 1j, +7 - 3j,
                          +7 + 7j, +7 + 5j, +7 + 1j, +7 + 3j,
                          +5 - 7j, +5 - 5j, +5 - 1j, +5 - 3j,
                          +5 + 7j, +5 + 5j, +5 + 1j, +5 + 3j,
                          +1 - 7j, +1 - 5j, +1 - 1j, +1 - 3j,
                          +1 + 7j, +1 + 5j, +1 + 1j, +1 + 3j,
                          +3 - 7j, +3 - 5j, +3 - 1j, +3 - 3j,
                          +3 + 7j, +3 + 5j, +3 + 1j, +3 + 3j,
                          ])
    qam64_arr = qam64_arr * np.sqrt(len(qam64_arr)) / np.linalg.norm(qam64_arr)
    return qam64_arr, 6

def get_constellation_point_index(mapper_arr, point):
    diff = np.abs(mapper_arr - point)
    return np.argmin(diff)

def demodulator(points, mapper_arr, bits_depth):
    indices = [get_constellation_point_index(point, mapper_arr) for point in points]
    bits_arr = []
    for i, ind in enumerate(indices):
        bits_arr.extend([int(digit) for digit in bin(ind)[2:].rjust(bits_depth, '0')])

    return bits_arr

def modulator(bits_arr, mapper_arr, bits_depth):
    bits_str = "".join([str(x) for x in bits_arr])
    indices = [int(bits_str[i:i + bits_depth], 2) for i in range(0, len(bits_arr), bits_depth)]
    points = [mapper_arr[index] for index in indices]
    return points

# def test_64_qam():
#     powered = qam64_arr_create()
#     for i in range(64):
#         if powered[i] != slower_qam64(i):
#             print(f"Comparison failed on {i}: got {powered[i]}, expected {slower_qam64(i)}")
# 
# def print_gray():
#     arr = []
#     for i in range(64):
#         arr.append(gray_code(i))
# 
#     to_print = ""
#     for i in range(8):
#         to_print_line = ""
#         for j in range(8):
#             to_print_line += f"{bin(arr[8 * i + j]).ljust(8)} "
#         to_print += to_print_line + "\n"
#     print(to_print)

def eval_ber(sent_bits, recv_bits):
    return np.sum(np.logical_xor(sent_bits, recv_bits)) / len(sent_bits)

def gen_awgn(points, snr):
    points_amount = len(points)
    root_points_power = np.linalg.norm(points) / np.sqrt(points_amount)
    # print(10 * np.log10(root_points_power ** 2))
    print(root_points_power)
    noise_power = root_points_power / (2 * (10 ** (snr / 10)))
    real_part = np.random.normal(scale = np.sqrt(noise_power), size=points_amount)
    imag_part = np.random.normal(scale = np.sqrt(noise_power), size=points_amount)

    return real_part + 1j * imag_part


def get_ess_table():
    return [ [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1],
             [0, 0, 0, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [0, 0, 0, 1, 1],
             [0, 0, 1, 0, 1], ]

def get_ccdm_table():
    return [ [0, 0, 0, 1, 1],
             [0, 0, 1, 0, 1],
             [0, 0, 1, 1, 0],
             [0, 1, 0, 0, 1],
             [0, 1, 0, 1, 0],
             [0, 1, 1, 0, 0],
             [1, 0, 0, 0, 1],
             [1, 0, 0, 1, 0], ]

def deshape_bits(bits, table):
    bits_amount = len(bits)
    output_bits = np.zeros(int(bits_amount * 3 / 5), dtype=int)
    for i in range(0, bits_amount, 20):
        phase_bits = bits[i : i + 20 : 2][:6]

        ampl_bits = bits[i + 1: i + 20 : 2]
        ampl_bits = np.logical_not(ampl_bits).astype(int)
        first_triplet_mapped = list(ampl_bits[:5])
        second_triplet_mapped = list(ampl_bits[5:])

        first_triplet = np.argmin(np.sum(np.abs(np.array(table) - first_triplet_mapped), axis=1))
        second_triplet = np.argmin(np.sum(np.abs(np.array(table) - second_triplet_mapped), axis=1))
        # first_triplet = table.index(first_triplet_mapped)
        # second_triplet = table.index(second_triplet_mapped)

        first_three_list = [int(digit) for digit in bin(first_triplet)[2:].rjust(3, '0')]
        second_three_list = [int(digit) for digit in bin(second_triplet)[2:].rjust(3, '0')]

        deshaped_ampl_bits = np.concat((first_three_list, second_three_list))

        output_index = int (i * 3 / 5)

        output_bits[output_index : output_index + 12 : 2] = phase_bits
        output_bits[output_index + 1 : output_index + 12 : 2] = deshaped_ampl_bits
    return output_bits


def shape_bits(bits, table):
    ampl_bits = bits[1::2]
    phase_bits = bits[0::2]
    ampl_bits_amount = len(ampl_bits)
    new_bits = np.zeros(int(ampl_bits_amount * 10 / 3), dtype=int)
    for i in range(0, ampl_bits_amount, 6):
        new_ampl_bits = []
        new_phase_bits = phase_bits[i : i + 6]
        gen_phase_bits = (np.random.rand(4) > 0.5).astype(int)
        new_phase_bits = np.concat((new_phase_bits, gen_phase_bits))

        triplet_first = int("".join(str(digit) for digit in ampl_bits[i : i + 3]), 2)
        triplet_second = int("".join(str(digit) for digit in ampl_bits[i + 3 : i + 6]), 2)

        new_ampl_bits.extend(table[triplet_first])
        new_ampl_bits.extend(table[triplet_second])

        new_ampl_bits = np.logical_not(new_ampl_bits).astype(int)

        new_bits_index = int(i * 20 / 6)
        new_bits[new_bits_index : new_bits_index + 20 : 2] = new_phase_bits
        new_bits[new_bits_index + 1 : new_bits_index + 20 : 2] = new_ampl_bits
        # breakpoint()

    return new_bits


def single_const_full(bits_amount, mod_create_fun, snr_range, theory_ber_fun, constellation_name):
    _, bit_depth = mod_create_fun()

    Eb_n0_dB = snr_range - 10 * np.log10(bit_depth)
    Eb_n0 = np.pow(10, Eb_n0_dB / 10)
    print(Eb_n0_dB)
    ber_arr = np.zeros(len(snr_range))
    true_ber_arr = [ theory_ber_fun(eb_n0_val) for eb_n0_val in Eb_n0]
    # print(true_ber_arr)
    for i, snr in enumerate(snr_range):
        print(i)
        ber_arr[i] = experiment(bits_amount, mod_create_fun, snr)
    # print(ber_arr)
    plt.figure()
    plt.plot(snr_range, ber_arr, label = 'practice')
    plt.plot(snr_range, true_ber_arr, label = 'theory')
    plt.xlabel('SNR [dB]')
    plt.yscale('log')
    plt.title(f"Theory vs modeling comparison for {constellation_name}")
    plt.grid()
    plt.ylabel('BER')
    plt.legend(fontsize = 10)
    plt.savefig(f"{constellation_name}_ther_model.png")
    # plt.show()

def experiment(bits_amount, mod_create_fun, snr):
    bits = (np.random.rand(bits_amount) > 0.5).astype(int)
     
    mod_arr, bit_depth = mod_create_fun()
    points = modulator(bits, mod_arr, bit_depth)

    noisy_points = points + gen_awgn(points, snr)
    # breakpoint()

    recv_bits = demodulator(noisy_points, mod_arr, bit_depth)
    ber = eval_ber(bits, recv_bits) 

    return ber

def qam16_bit_shaping(bits_amount, snr, table_fun):
    bits = (np.random.rand(bits_amount) > 0.5).astype(int)
    # print(f"started with {bits}, ampl: {bits[1::2]}")
    new_bits = shape_bits(bits, table_fun())
    mods = modulator(new_bits, *qam16_arr_create())
    noisy = mods + gen_awgn(mods, snr)
    demodulated = demodulator(noisy, *qam16_arr_create())
    # print(f"got {new_bits}, ampls notted: {np.logical_not(new_bits[1::2]).astype(int)}")
    recv = deshape_bits(demodulated, table_fun())
    # breakpoint()
    return eval_ber(demodulated, new_bits)

def show_constellations(mapper_arr, bit_depth, constellation_name):
    point_amount = int(2 ** bit_depth)
    indices = range(point_amount)
    indices_in_text = [bin(val)[2:].rjust(bit_depth, '0') for val in indices]
    points_x = [np.real(point) for point in mapper_arr]
    points_y = [np.imag(point) for point in mapper_arr]
    plt.figure()
    plt.scatter(points_x, points_y)
    for x, y, name in zip(points_x, points_y, indices_in_text):
        plt.annotate(name, (x, y))
    plt.grid()
    plt.title(f"{constellation_name.upper()} points")
    plt.savefig(f"{constellation_name}.png")

def compare_shaping_res():
    snr_range = np.arange(0, 20)
    snr_amount = len(snr_range)
    ber_shaped_ess = np.zeros(snr_amount)
    ber_shaped_ccdm = np.zeros(snr_amount)
    ber_uniformed = np.zeros(snr_amount)
    bits_amount = 240000
    for i, snr in enumerate(snr_range):
        print(i)
        ber_shaped_ess[i] = qam16_bit_shaping(bits_amount, snr, get_ess_table)
        ber_shaped_ccdm[i] = qam16_bit_shaping(bits_amount, snr, get_ccdm_table)
        ber_uniformed[i] = experiment(bits_amount, qam16_arr_create, snr)

    plt.plot(snr_range, ber_shaped_ess, label = 'ESS shaping')
    plt.plot(snr_range, ber_shaped_ccdm, label = 'CCDM shaping')
    plt.plot(snr_range, ber_uniformed, label = 'Uniform')
    plt.xlabel('SNR [dB]')
    plt.yscale('log')
    plt.grid()
    plt.ylabel('BER')
    plt.title('BER(SNR) for different shaping for QAM16 comparison')
    plt.legend(fontsize = 10)
    plt.savefig("with_shaping.png")
    plt.show()

def compare_theory_practice():
    funs = [qpsk_arr_create, qam16_arr_create, qam64_arr_create]
    snr_ranges = [np.arange(0, 14), np.arange(0, 20), np.arange(0, 20)]
    theory_funcs = [lambda x : math.erfc(math.sqrt(x)) / 2,
                    lambda x : math.erfc(math.sqrt(x * 2 / 5)) * 3 / 8,
                    lambda x : math.erfc(math.sqrt(3 * x / 21)) * (1 - 1 / 8) / 3,
                    ]
    names = ["QPSK", "QAM16", "QAM64"]
    for i in range(0, len(snr_ranges)):
        single_const_full(12 * 100000, funs[i], snr_ranges[i], theory_funcs[i], names[i])

    # print_gray()
    # print(powered)

def get_constellation_graphs():
    show_constellations(*qpsk_arr_create(), "qpsk")
    show_constellations(*qam16_arr_create(), "qam16")
    show_constellations(*qam64_arr_create(), "qam64")

if __name__ == "__main__":
    # print(test_bit_shaping(240000, 30))
    # compare_theory_practice()
    # get_constellation_graphs()
    compare_shaping_res()
