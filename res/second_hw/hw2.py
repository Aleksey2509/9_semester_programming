import scipy
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
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
        # fun_to_integr = partial(for_integr, k_val=k, l=l)
        res, err = scipy.integrate.quad(for_integr, 0, np.pi, args=(k, l))
        res = 60 * res
        ln_memb = np.log(l / a) - 1
        alpha = res / (120 * l * ln_memb * (1 - np.sin(2 * k * l) / (2 * k * l)))
        Z_b = 120 * ln_memb * (1 - 1j * alpha / beta)
        gamma = alpha + 1j * beta
        Z_arr[i] = Z_b / np.tanh(gamma)
        asdasdasdasdsa = 1
    return nu_values, Z_arr


def task1():
    plt.ion()
    nu_values, Z_arr = get_impedance()
    admittance = 1 / Z_arr
    plt.figure()
    plt.plot(nu_values, np.real(Z_arr))
    plt.xlabel("frequencies (MHZ)")
    plt.ylabel("Real part of impedance")
    plt.title("Real(Z) (f)")
    plt.grid()
    plt.savefig(f"pictures/Real_Z.png")
    plt.show()

    plt.figure()
    plt.plot(nu_values, np.imag(Z_arr))
    plt.xlabel("frequencies (MHZ)")
    plt.ylabel("Imaginary part of impedance")
    plt.title("Imag(Z) (f)")
    plt.grid()
    plt.savefig(f"pictures/Imag_Z.png")
    plt.show()
    
    plt.figure()
    plt.plot(nu_values, np.real(admittance))
    plt.xlabel("frequencies (MHZ)")
    plt.ylabel("Real part of admittance")
    plt.title("Real(Y) (f)")
    plt.grid()
    plt.savefig(f"pictures/Real_Y.png")
    plt.show()

    plt.figure()
    plt.plot(nu_values, np.imag(admittance))
    plt.xlabel("frequencies (MHZ)")
    plt.ylabel("Imaginary part of admittance")
    plt.title("Imag(Y) (f)")
    plt.grid()
    plt.savefig(f"pictures/Imag_Y.png")
    plt.show()

    gamma_arr = (Z_arr - 50) / (Z_arr + 50)
    plt.figure()
    plt.plot(nu_values, np.abs(gamma_arr))
    plt.xlabel("frequencies (MHZ)")
    plt.ylabel("|gamma|")
    plt.title("Absolute value of reflection coefficient (f)")
    plt.savefig(f"pictures/Gamma1.png")
    plt.grid()
    plt.show()

    gamma_arr_dB = 10 * np.log10(np.abs(gamma_arr))
    plt.figure()
    plt.plot(nu_values, gamma_arr_dB)
    plt.xlabel("frequencies (MHZ)")
    plt.ylabel("|gamma|(dB)")
    plt.title("Absolute value of reflection coefficient in dB (f)")
    plt.savefig(f"pictures/Gamma1_dB.png")
    plt.grid()
    plt.show()
    
    plt.figure()
    ax = plt.subplot(111, projection="polar")
    ax.plot(np.angle(gamma_arr), np.abs(gamma_arr))
    plt.xlabel('Real(gamma)')
    plt.ylabel('Imag(gamma)')
    plt.title('Gamma')
    plt.savefig(f"pictures/Gamma1_polar.png")
    plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "smith"})
    ax.plot(Z_arr, datatype="Z")
    plt.title("Smith diagramm of impedance")
    plt.savefig(f"pictures/smith_z50.png")
    plt.show()
    
    ax.update_scParams(axes_impedance=75)
    fig, ax = plt.subplots(subplot_kw={"projection": "smith"})
    ax.plot(Z_arr, datatype="Z")
    plt.title("Smith diagramm of impedance")
    plt.savefig(f"pictures/smith_z75.png")
    plt.show()


def get_resist_with_line(Z_0, nu_0=100):
    nu_values, Z_arr = get_impedance()
    nu_0_index = np.where(np.isclose(nu_values, nu_0))[0][0]
    omega_C = 0.0
    omega_L = 0.0
    Y_0 = 1 / Z_0

    Z = Z_arr[nu_0_index]
    R_L = np.real(Z)
    X_L = np.imag(Z)
    norm_Z = np.abs(Z) * np.abs(Z)
    G_L = np.real(1 / Z)
    B_L = np.imag(1 / Z)

    if R_L > Z_0:
        common_part = np.sqrt((Y_0 - G_L) * G_L)
        true_C = (common_part - B_L) / (2 * np.pi * nu_0 * 1e6)
        true_L = (common_part / (Y_0 * G_L)) / (2 * np.pi * nu_0 * 1e6)

        omega_C = (common_part - B_L) * nu_values / nu_0
        omega_L = (common_part / (Y_0 * G_L)) * nu_values / nu_0
        equiv_Z = 1j * omega_L + Z_arr / (1 + 1j * omega_C * Z_arr)
    else:
        common_part = np.sqrt((Z_0 - R_L) * R_L)
        omega_L = (common_part - X_L) * nu_values / nu_0
        omega_C = (common_part / (Z_0 * R_L)) * nu_values / nu_0
        resist_sum = Z + 1j * omega_L[i]
        equiv_Z = resist_sum / (1 + 1j * omega_C * resist_sum)

    return nu_values, Z_arr, omega_C, omega_L, equiv_Z


def task2(Z_0=50):
    nu_values, Z_arr, omega_C, omega_L, equiv_Z = get_resist_with_line(Z_0)
    # print(nu_values)
    # print(equiv_Z)
    plt.ion()

    plt.figure()
    plt.plot(nu_values, np.real(equiv_Z))
    plt.xlabel("frequencies (MHZ)")
    plt.ylabel("Real part of impedance after matching")
    plt.title("Real(Z_matched) (f)")
    plt.grid()
    plt.savefig(f"pictures/Real_Z_matched.png")
    plt.show()

    plt.figure()
    plt.plot(nu_values, np.imag(equiv_Z))
    plt.xlabel("frequencies (MHZ)")
    plt.ylabel("Imaginary part of impedance after matching")
    plt.title("Imag(Z_matched) (f)")
    plt.grid()
    plt.savefig(f"pictures/Imag_Z_matched.png")
    plt.show()

    gamma_arr = (equiv_Z - Z_0) / (equiv_Z + Z_0)
    plt.figure()
    plt.plot(nu_values, np.abs(gamma_arr))
    plt.xlabel("frequencies (MHZ)")
    plt.ylabel("|gamma|")
    plt.title("Absolute value of reflection coefficient (f)")
    plt.grid()
    plt.savefig(f"pictures/Gamma2.png")
    plt.show()

    gamma_arr_dB = 10 * np.log10(np.abs(gamma_arr))
    plt.figure()
    plt.plot(nu_values, gamma_arr_dB)
    plt.xlabel("frequencies (MHZ)")
    plt.ylabel("|gamma| (dB)")
    plt.title("Absolute value of reflection coefficient in dB (f)")
    plt.grid()
    plt.savefig(f"pictures/Gamma2_dB.png")
    plt.show()

    SWR = (1 + np.abs(gamma_arr)) / (1 - np.abs(gamma_arr))
    plt.figure()
    plt.plot(nu_values, SWR)
    plt.xlabel("frequencies (MHZ)")
    plt.ylabel("SWR")
    plt.title("Standing wave ratio(f)")
    plt.grid()
    plt.savefig(f"pictures/SWR.png")
    plt.show()

    plt.figure()
    ax = plt.subplot(111, projection="polar")
    ax.plot(np.angle(gamma_arr), np.abs(gamma_arr))
    plt.xlabel('Real(gamma)')
    plt.ylabel('Imag(gamma)')
    plt.title('Gamma')
    plt.savefig(f"pictures/Gamma2_polar.png")
    plt.show()

    return


def get_noise(F_array, K_array):
    inverted_K = jnp.divide(jnp.ones(K_array.size), K_array)
    inverted_K = jnp.roll(inverted_K, 1)
    inverted_K = inverted_K.at[0].set(1)
    inverted_K_cumprod = jnp.cumprod(inverted_K)

    F_array_minus_one = F_array - jnp.ones(F_array.size)
    F_array_minus_one = F_array_minus_one.at[0].set(F_array[0])
    return jnp.sum(F_array_minus_one * inverted_K_cumprod)


def get_IP3(IP3_arr, K_array):
    inverted_K = jnp.divide(jnp.ones(K_array.size), K_array)
    inverted_K = jnp.roll(inverted_K, 1)
    inverted_K = inverted_K.at[0].set(1)
    inverted_K_cumprod = jnp.cumprod(inverted_K)

    IP3_for_cumsum = jnp.divide(
        jnp.ones(IP3_arr.size), jnp.square(IP3_arr * inverted_K_cumprod)
    )

    IP3_cumsum = jnp.sqrt(
        jnp.cumsum(jnp.divide(jnp.ones(IP3_for_cumsum.size), IP3_for_cumsum))
    )
    return IP3_cumsum[-1]


def task3():
    K_arr_dB = np.array([-3.5, 10, -3.5, 13, -3.1, 14.1, 0])
    K_arr = np.power(10, K_arr_dB / 10)
    F_arr_dB = np.array([3.5, 0.9, 3.5, 9, 3.1, 1.7, 29.5])
    F_arr = np.power(10, F_arr_dB / 10)
    IP3_arr_dB = np.array([100, 14.5, 100, 9.8, 100, 20, 30.5])
    IP3_arr = np.power(10, IP3_arr_dB / 10)

    K_arr_to_cum_prod = 1 / np.concatenate(([1], K_arr))[:-1]
    K_cumprod = np.cumprod(K_arr_to_cum_prod)
    F_minus_one = F_arr - 1
    F_minus_one[0] = F_arr[0]

    noise_grad = jax.grad(get_noise, argnums=0)
    print(f"noise grads: {noise_grad(jnp.array(F_arr), jnp.array(K_arr))}")
    ip3_grad = jax.grad(get_IP3, argnums=0)
    print(f"ip3 grads: {ip3_grad(jnp.array(IP3_arr), jnp.array(K_arr))}")

    noise_factors_to_cum_sum = F_minus_one * K_cumprod
    noise_factors = np.cumsum(F_minus_one * K_cumprod)
    print(f"noise factors singled {noise_factors_to_cum_sum}")
    print(f"noise factors {noise_factors}")

    IP3_for_cumsum = 1 / (IP3_arr * K_cumprod) ** 2
    IP3_cumsum = np.sqrt(1 / np.cumsum(IP3_for_cumsum))
    print(f"IP3 before cumsum : {IP3_for_cumsum}")
    print(f"IP3: {IP3_cumsum}")

if __name__ == "__main__":
    task1()
    task2()
    input('Hey')
