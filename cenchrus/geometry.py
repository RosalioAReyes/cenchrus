import numpy as np
import shapely
from shapely.geometry import MultiPoint

def concave_mask(matrix, ratio):
    # input: binarized matrix 
    y_coords, x_coords = np.nonzero(matrix > 0)
    puntos = np.column_stack((x_coords, y_coords))
    coleccion_puntos = MultiPoint(puntos)
    envoltura_concava = shapely.concave_hull(coleccion_puntos, ratio=ratio)
    x, y = envoltura_concava.exterior.xy
    nfilas, ncolumnas = matrix.shape
    r_grid, c_grid = np.meshgrid(np.arange(nfilas), np.arange(ncolumnas), indexing='ij')
    #dentro_poligono = shapely.contains_xy(envoltura_concava, c_grid.ravel(), r_grid.ravel()) # extrictely inside
    dentro_poligono =shapely.intersects_xy(envoltura_concava, c_grid.ravel(), r_grid.ravel()) # intersects (includes boundary)
    mascara_envoltura = dentro_poligono.reshape((nfilas, ncolumnas))
    return mascara_envoltura, x, y

from shapely.geometry import Polygon

def union_concave_masks(xp, yp, xs, ys, matrix_shape):
    poly_planted = Polygon(zip(xp, yp))
    poly_self = Polygon(zip(xs, ys))
    combined_polygon = shapely.unary_union([poly_planted, poly_self])
    xa, ya = combined_polygon.exterior.xy
    
    nrows, ncols = matrix_shape
    r_grid, c_grid = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing='ij')
    dentro_poligono = shapely.intersects_xy(combined_polygon, c_grid.ravel(), r_grid.ravel())
    union_mask = dentro_poligono.reshape((nrows, ncols))
    return union_mask, xa, ya
