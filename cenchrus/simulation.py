from scipy.stats import lognorm
from .kernels import kernel_convolution
from .analysis import analyze_patches
import numpy as np
def compute_arrival_probability(Nt, kernel, nrows, ncols):
    return 1.0 - kernel_convolution(Nt, kernel, ncols, nrows)

def select_colonization_sites(P_arrival, th1, th2, rng):
    eligible = np.nonzero(P_arrival > th1)
    if eligible[0].size == 0:
        return None
    accepted = rng.random(eligible[0].size) < th2
    if not np.any(accepted):
        return None
    return (eligible[0][accepted], eligible[1][accepted],)

def generate_new_biomass(current_biomass, carrying_capacity, s, loc, scale, rng):
    potential = lognorm.rvs( s=s, loc=loc, scale=scale, size=current_biomass.size, random_state=rng,)
    space_fraction = np.clip( 1.0 - current_biomass / carrying_capacity, 0.0, 1.0,)
    return potential * space_fraction

def colonization_step(Nt, kernel, params, rng):
    th1, th2, K, s, loc, scale, nrows, ncols = params
    Nt_new = Nt.copy()
    P = compute_arrival_probability(Nt, kernel, nrows, ncols,)
    coords = select_colonization_sites(P, th1, th2, rng, )

    if coords is None:
        return Nt_new

    biomass = generate_new_biomass(Nt_new[coords], K, s, loc, scale, rng,)
    Nt_new[coords] += biomass
    return Nt_new

def simulate(planted, Kernel2convol, params, t_pasos, seed):
    rng = np.random.default_rng(seed)
    M_i = colonization_step(planted, Kernel2convol, params, rng)
    for _ in range(t_pasos):
        M_i = colonization_step(M_i, Kernel2convol, params, rng)
    M_i = M_i - planted
    return M_i

def simulate_with_source(planted, Kernel2convol, params, t_pasos, seed):
    rng = np.random.default_rng(seed)
    M_i = colonization_step(planted, Kernel2convol, params, rng)
    for _ in range(t_pasos):
        M_i = colonization_step(M_i, Kernel2convol, params, rng)
    return M_i

def introduction_step(M, n_intro, rng, intro_size):
    M_new = M.copy()
    empty = np.argwhere(M_new == 0)
    n_intro = min(n_intro, len(empty))
    idx = rng.choice(len(empty), n_intro, replace=False)
    new_sites = empty[idx]
    M_new[new_sites[:, 0], new_sites[:, 1]] = intro_size
    return M_new

def simulate_with_introduction(planted, Kernel2convol, params, t_pasos, seed, n_intro, intro_size):
    # with source
    rng = np.random.default_rng(seed)
    M_i = planted.copy()
    for _ in range(t_pasos):
        M_i = introduction_step(M_i, n_intro, rng, intro_size)
        M_i = colonization_step(M_i, Kernel2convol, params, rng)
    return (M_i > 0).astype(np.uint32)

def simulate_statistics(planted, Kernel2convol, params, t_pasos, seed, mascara_envoltura):
    """
    Runs a single simulation, masks it, analyzes patches, and returns the stats.
    """
    rng = np.random.default_rng(seed)
    # 1. Run the time steps
    M_i = colonization_step(planted, Kernel2convol, params, rng)
    for _ in range(t_pasos):
        M_i = colonization_step(M_i, Kernel2convol, params, rng)
    M_i = M_i - planted
    # 2. Apply the mask
    M_i_hull = np.where(mascara_envoltura, M_i, 0)
    # 3. Analyze the patches 
    stats = analyze_patches(M_i_hull, plot=False)
    return stats

from concurrent.futures import ProcessPoolExecutor
import numpy as np


def simulate_histogram(planted, Kernel2convol, params,
                       t_pasos, seed, mascara_envoltura, bins):
    """
    Intensity pixel
    Runs a single simulation, masks it, and returns the normalized histogram.
    """
    rng = np.random.default_rng(seed)
    M_i = colonization_step(planted, Kernel2convol, params, rng)
    for _ in range(t_pasos):
        M_i = colonization_step(M_i, Kernel2convol, params, rng)

    M_i = M_i - planted
    M_i_hull = np.where(mascara_envoltura, M_i, 0)
    values = M_i_hull[M_i_hull > 0]
    hist, _ = np.histogram(values, bins=bins, density=True)
    return hist

def simulate_histograms(planted, Kernel2convol, params, t_pasos, mascara_envoltura, bins,  m_realizations):
    """
    Intensity pixel
    Runs multiple stochastic realizations in parallel and returns
    the mean and standard deviation of the histograms.
    """

    seeds = np.random.randint( 0, np.iinfo(np.int32).max, size=m_realizations,)

    with ProcessPoolExecutor() as executor:

        histograms = list( executor.map( simulate_histogram, [planted] * m_realizations, [Kernel2convol] * m_realizations, [params] * m_realizations, [t_pasos] * m_realizations, seeds, [mascara_envoltura] * m_realizations, [bins] * m_realizations, ) )

    histograms = np.asarray(histograms)
    hist_mean = np.mean(histograms, axis=0)
    hist_std = np.std(histograms, axis=0)
    return hist_mean, hist_std

from scipy.spatial.distance import pdist
from .raster import transform_to_coordinates

def simulate_distance_histogram(planted, Kernel2convol, params, t_pasos, seed, mascara_envoltura, transform, bins):
    rng = np.random.default_rng(seed)
    M_i = colonization_step(planted, Kernel2convol, params, rng)

    for _ in range(t_pasos):
        M_i = colonization_step(M_i, Kernel2convol, params, rng)

    M_i = M_i - planted
    M_i_hull = np.where(mascara_envoltura, M_i, 0)
    x, y, values = transform_to_coordinates(M_i_hull, transform)
    active = values > 0
    points = np.column_stack((x[active], y[active]))
    distances = pdist(points)
    hist, _ = np.histogram(distances, bins=bins, density=True)

    return hist

def simulate_distance_histograms(planted, Kernel2convol, params, t_pasos, mascara_envoltura, transform, bins, m_realizations):

    seeds = np.random.randint(0, np.iinfo(np.int32).max, size=m_realizations)

    with ProcessPoolExecutor() as executor:
        histograms = list(executor.map(simulate_distance_histogram, [planted]*m_realizations, [Kernel2convol]*m_realizations, [params]*m_realizations, [t_pasos]*m_realizations, seeds, [mascara_envoltura]*m_realizations, [transform]*m_realizations, [bins]*m_realizations))

    histograms = np.asarray(histograms)
    return histograms.mean(axis=0), histograms.std(axis=0)


# para calcular probabilidades de infección
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def simulate_infection(planted, Kernel2convol, params, t_pasos, seed):
    rng = np.random.default_rng(seed)
    M_i = colonization_step(planted, Kernel2convol, params, rng)
    for _ in range(t_pasos):
        M_i = colonization_step(M_i, Kernel2convol, params, rng)

    #M_i = M_i - planted.   important, because if we subtract planted, we do not consider initial biomass
    return (M_i > 0).astype(np.uint32)


def simulate_infection_probability(planted, Kernel2convol, params, t_pasos, m_realizations):

    seeds = np.random.randint(0, np.iinfo(np.int32).max, size=m_realizations)
    counts = np.zeros(planted.shape, dtype=np.uint32)

    with ProcessPoolExecutor() as executor:
        for infection in executor.map(simulate_infection,[planted] * m_realizations,[Kernel2convol] * m_realizations,[params] * m_realizations,[t_pasos] * m_realizations,seeds,):
            counts += infection

    return counts / m_realizations

def simulate_infection_probability_with_introduction(planted,Kernel2convol,params,t_pasos,m_realizations,n_intro,intro_size):

    seeds = np.random.randint(0,np.iinfo(np.int32).max,size=m_realizations)
    counts = np.zeros(planted.shape, dtype=np.uint32)

    with ProcessPoolExecutor() as executor:
        for infection in executor.map(simulate_with_introduction,[planted] * m_realizations,[Kernel2convol] * m_realizations,[params] * m_realizations,[t_pasos] * m_realizations,seeds,[n_intro] * m_realizations,[intro_size] * m_realizations):
            counts += infection

    return counts / m_realizations