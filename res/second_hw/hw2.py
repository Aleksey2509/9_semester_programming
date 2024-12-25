import scipy
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from smithplot import SmithAxes


def for_integr(theta, k_val, l):
    numer = np.cos(k_val * l * np.cos(theta)) - np.cos(k_val * l)
    numer = numer * numer
    return numer / np.sin(theta)


def get_impedance():
    nu_values = np.arange(50, 500, 1)
    k_arr = 2 * np.pi * nu_values / 300
    Z_arr = np.zeros_like(k_arr, dtype=np.complex128)
    l = 1
    a = 0.001
    for i, k in enumerate(k_arr):
        beta = k
        fun_to_integr = partial(for_integr, k_val=k, l=l)
        res = scipy.integrate.quad(fun_to_integr, 0, np.pi)[0]
        res = 60 * res
        ln_memb = np.log(l / a) - 1
        alpha = res / (120 * ln_memb * (1 - np.sinc(2 * k * l)))
        Z_b = 120 * ln_memb * (1 - 1j * alpha / beta)
        gamma = alpha + 1j * beta
        Z_arr[i] = Z_b / np.tanh(gamma)
    return nu_values, Z_arr


def task1():
    nu_values, Z_arr = get_impedance()
    plt.figure()
    plt.plot(nu_values, np.real(Z_arr))
    plt.show()
    plt.figure()
    plt.plot(nu_values, np.imag(Z_arr))
    plt.show()

    gamma_arr = (Z_arr - 50) / (Z_arr + 50)
    plt.figure()
    plt.plot(nu_values, np.abs(gamma_arr))
    plt.show()

    gamma_arr_dB = 10 * np.log10(np.abs(gamma_arr))
    plt.figure()
    plt.plot(nu_values, gamma_arr_dB)
    plt.show()

    plt.figure()
    ax = plt.subplot(1, 1, 1, projection="smith")
    plt.plot([10, 100], markevery=1)
    plt.plot(Z_arr, datatype=SmithAxes.Z_PARAMETER)
    plt.show()


def get_resist_with_line(Z_0, nu_0=100):
    nu_values, Z_arr = get_impedance()
    omega_C = np.zeros_like(Z_arr, dtype=np.complex128)
    omega_L = np.zeros_like(Z_arr, dtype=np.complex128)
    equiv_Z = np.zeros_like(Z_arr, dtype=np.complex128)
    Y_0 = 1 / Z_0
    for i, Z in enumerate(Z_arr):
        R_L = np.real(Z)
        X_L = np.imag(Z)
        norm_Z = np.abs(Z) * np.abs(Z)
        G_L = R_L / norm_Z
        B_L = -X_L / norm_Z

        if R_L > Z_0:
            common_part = np.sqrt((Y_0 - G_L) * G_L)
            omega_C[i] = (common_part - B_L) * nu_values[i] / nu_0
            omega_L[i] = (common_part / (Y_0 * G_L)) * nu_values[i] / nu_0
            equiv_Z[i] = 1j * omega_L[i] + Z / (1 + 1j * omega_C[i] * Z)
        else:
            common_part = np.sqrt((Z_0 - R_L) * R_L)
            omega_L[i] = (common_part - X_L) * nu_values[i] / nu_0
            omega_C[i] = (common_part / (Z_0 * R_L)) * nu_values[i] / nu_0
            resist_sum = Z + 1j * omega_L[i]
            equiv_Z[i] = resist_sum / (1 + 1j * omega_C[i] * resist_sum)

    return nu_values, Z_arr, omega_C, omega_L, equiv_Z


def task2(Z_0=50):
    nu_values, Z_arr, omega_C, omega_L, equiv_Z = get_resist_with_line(Z_0)
    print(nu_values)
    print(equiv_Z)

    plt.figure()
    plt.plot(nu_values, np.real(equiv_Z))
    plt.show()

    plt.figure()
    plt.plot(nu_values, np.imag(equiv_Z))
    plt.show()

    gamma_arr = (equiv_Z - Z_0) / (equiv_Z + Z_0)
    plt.figure()
    plt.plot(nu_values, np.abs(gamma_arr))
    plt.show()

    gamma_arr_dB = 10 * np.log10(np.abs(gamma_arr))
    plt.figure()
    plt.plot(nu_values, gamma_arr_dB)
    plt.show()

    CSW = (1 + np.abs(gamma_arr)) / (1 - np.abs(gamma_arr))
    plt.figure()
    plt.plot(nu_values, CSW)
    plt.show()

    plt.figure()
    ax = plt.subplot(1, 1, 1, projection="smith")
    plt.plot([10, 100], markevery=1)
    plt.plot(gamma_arr, datatype=SmithAxes.Z_PARAMETER)
    plt.show()


def task3():
    K_arr_dB = np.array([-3.5, 10, -3.5, 13, -3.1, 14.1, 0])
    K_arr = np.power(10, K_arr_dB / 10)
    F_arr_dB = np.array([3.5, 0.9, 3.5, 9, 3.1, 1.7, 29.5])
    F_arr = np.power(10, F_arr_dB / 10)
    IP3_arr_dB = np.array([100, 14.5, 100, 9.8, 100, 20, 30.5])
    IP3_arr = np.power(10, IP3_arr_dB / 10)

    K_arr_to_cum_prod = 1 / np.concatenate(([1], K_arr))[:-1]
    print(K_arr_to_cum_prod)
    K_cumprod = np.cumprod(K_arr_to_cum_prod)
    F_minus_one = F_arr - 1
    F_minus_one[0] = F_arr[0]

    IP3_for_cumsum = 1 / np.square(IP3_arr * K_arr_to_cum_prod)
    IP3_cumsum = np.sqrt(1 / np.cumsum(IP3_for_cumsum)) 

    noise_factors = np.cumsum(F_minus_one * K_cumprod)
    print(f"noise factors {noise_factors}")
    print(f"IP3: {IP3_cumsum}")


if __name__ == "__main__":
    task3()
