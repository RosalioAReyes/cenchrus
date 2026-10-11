import numpy as np
import rasterio
def transform_to_coordinates(M, transform):
    rows, cols = np.indices(M.shape)
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
    return np.asarray(xs).ravel(), np.asarray(ys).ravel(), M.ravel()