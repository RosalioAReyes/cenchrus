import numpy as np
from joblib import Parallel, delayed
from .simulation import simulate_statistics
from .kernels import exp_kernel_fft
def objective_function(params_opt, stats_self,planted,params_extra, t_pasos, mascara_envoltura):
        alpha, th, th2, s, scale = params_opt 
        carrying_capacity, loc, nrows, ncols, m_realizations, pixel_size = params_extra
        params = th, th2, carrying_capacity,  s, loc, scale, nrows, ncols
        Kernel2convol = exp_kernel_fft(ncols, nrows, alpha, pixel_size)
        results = Parallel(n_jobs=-1)(delayed(simulate_statistics)(planted, Kernel2convol, params, t_pasos, i, mascara_envoltura) for i in range(m_realizations))
        stats_array = np.array(results, dtype=float)
        average_stats = np.nanmean(stats_array, axis=0)
        # num_patches, area_mean, area_std, dist_mean, dist_std, intensity_mean, intensity_std
        weights = np.array([0.8, 0.15, 0, 0.025, 0,  0.025, 0])
        d = np.sum(weights * (stats_self -  average_stats)**2)
        return d
