import numpy as np
from scipy.signal import oaconvolve

from .simulation import (select_colonization_sites, generate_new_biomass)


def exp_kernel_mexico(alpha, pixel_size, radius_km):
    """
    Kernel exponencial truncada para la simulación nacional.
    """
    radius_pixels = int(np.ceil(radius_km / pixel_size))
    di = np.arange(-radius_pixels, radius_pixels + 1,)
    dj = np.arange(-radius_pixels, radius_pixels + 1,)
    DI, DJ = np.meshgrid(di, dj, indexing="ij")
    D = pixel_size * np.sqrt(DI**2 + DJ**2)
    K = ((alpha**2) / (2 * np.pi) * np.exp(-alpha * D))
    return K.astype(np.float32)



def kernel_convolution_mexico(M, K, ncols, nrows):
    """
    Convolución utilizada exclusivamente por la simulación nacional.
    Utiliza oaconvolve porque M es muy grande y K es pequeña.
    """

    lam = oaconvolve(M, K, mode="same")
    return np.exp(-lam)


def compute_arrival_probability_mexico(Nt, kernel, nrows, ncols):
    """
    Calcula la probabilidad de llegada en cada pixel.
    """
    return 1.0 - kernel_convolution_mexico(Nt, kernel, ncols, nrows)


# PASO DE COLONIZACIÓN
def colonization_step_mexico(Nt, kernel, params, rng):

    th1, th2, K, s, loc, scale, nrows, ncols = params
    Nt_new = Nt.copy()
    P = compute_arrival_probability_mexico(Nt, kernel, nrows, ncols)
    coords = select_colonization_sites(P, th1, th2, rng)

    if coords is None:
        return Nt_new

    biomass = generate_new_biomass(Nt_new[coords], K, s, loc, scale, rng)
    Nt_new[coords] += biomass
    return Nt_new


# INTRODUCCIÓN
def introduction_step_mexico(M, n_intro, rng, intro_size):

    M_new = M.copy()
    if n_intro <= 0:
        return M_new

    empty = np.argwhere(M_new == 0)

    if len(empty) == 0:
        return M_new

    n_intro = min(n_intro, len(empty))
    idx = rng.choice(len(empty), n_intro, replace=False)
    new_sites = empty[idx]
    M_new[new_sites[:, 0], new_sites[:, 1]] = intro_size

    return M_new


# UNA REALIZACIÓN
def simulate_infection_mexico(planted, Kernel2convol, params, t_pasos, seed, n_intro, intro_size):

    rng = np.random.default_rng(seed)
    M_i = planted.copy()

    for _ in range(t_pasos):
        M_i = introduction_step_mexico(M_i, n_intro, rng, intro_size)
        M_i = colonization_step_mexico(M_i, Kernel2convol, params, rng)

    return (M_i > 0).astype(np.uint32)


# PROBABILIDAD PROMEDIO
def simulate_infection_probability_mexico(planted, Kernel2convol, params, t_pasos, m_realizations, n_intro, intro_size):

    seeds = np.random.randint(0, np.iinfo(np.int32).max, size=m_realizations)
    counts = np.zeros(planted.shape, dtype=np.uint32)

    for i, seed in enumerate(seeds):

        print(f"Realization {i + 1}/{m_realizations}")
        infection = simulate_infection_mexico(planted, Kernel2convol, params, t_pasos, seed, n_intro, intro_size)
        counts += infection

        del infection

    probability = counts.astype(np.float32) / m_realizations
    return probability