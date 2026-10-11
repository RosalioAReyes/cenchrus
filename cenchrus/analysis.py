import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure

def analyze_patches(matrix, plot=False):
    """
    Analyzes a binary matrix using 8-connectivity (horizontal, vertical and diagonal) to find patches, computes
    area and distance statistics, and optionally plots the results.
    
    Returns: the vector of statistics: [num_patches, area_mean, area_std, distance_mean, distance_std]
    """
    intensity_mean = np.mean(matrix[matrix > 0]) if np.any(matrix > 0) else 0.0
    intensity_std = np.std(matrix[matrix > 0]) if np.any(matrix > 0) else 0.0
    matrix = np.where(matrix > 0, 1, 0)  # Binarize the matrix
    # 1. Label the patches using an 8-connectivity structural element
    structure = np.ones((3, 3))
    labeled_mask, num_patches = ndimage.label(matrix, structure=structure)
    
    # Initialize statistics
    area_mean = area_std = 0.0
    dist_mean = dist_std = None
    
    if num_patches > 0:
        # 2. Calculate patch areas
        patch_ids = np.arange(1, num_patches + 1)
        areas = ndimage.sum_labels(matrix, labeled_mask, index=patch_ids)
        area_mean = float(np.mean(areas))
        area_std = float(np.std(areas))
        
        # 3. Calculate distances between centroids (centers of mass)
        # ndimage.center_of_mass returns a list of (row, col) tuples
        centroids = ndimage.center_of_mass(matrix, labeled_mask, index=patch_ids)
        centroids = np.array(centroids)  # Shape: (num_patches, 2)
        
        if num_patches >= 2:
            # Compute a pairwise Euclidean distance matrix between all centroids
            diff = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
            dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
            
            # Extract only the upper triangle indices to avoid self-distances (0) and duplicates
            triu_indices = np.triu_indices(num_patches, k=1)
            unique_distances = dist_matrix[triu_indices]
            
            dist_mean = float(np.mean(unique_distances))
            dist_std = float(np.std(unique_distances))
            
    # 4. Compile the statistics dictionary
    stats = np.array([num_patches, area_mean, area_std, dist_mean, dist_std, intensity_mean, intensity_std], dtype=float)
    
    # 5. Plotting if requested
    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        # Display the binary heatmap
        im = ax.imshow(matrix, cmap='YlOrRd', origin='upper', interpolation='nearest')
        # Loop through each patch and extract boundary contours
        for patch_id in range(1, num_patches + 1):
            single_patch_mask = (labeled_mask == patch_id).astype(float)
            contours = measure.find_contours(single_patch_mask, level=0.5)
            for contour in contours:
                # contour is in (row, col) format, map col to X and row to Y
                ax.plot(contour[:, 1], contour[:, 0], color='blue', linewidth=2.5)
            # Optional: Plot a small blue dot at the centroid of each patch
            cy, cx = centroids[patch_id - 1]
            ax.plot(cx, cy, 'bo', markersize=5)
        ax.set_title(f"Heatmap Overlay (Found {num_patches} Patches)", fontsize=14)
        plt.colorbar(im, label="Matrix Value", ax=ax, shrink=0.7)
        plt.tight_layout()
        plt.show()
        
    return stats
