import numpy as np
                                                                                                             
def add_gaussian_noise(matrix, sensitivity, epsilon, delta):
    """
    Add Gaussian noise to a matrix to achieve differential privacy.

    Args:
        matrix (numpy.ndarray): The input matrix to which noise will be added.
        sensitivity (float): The sensitivity of the matrix.
        epsilon (float): The privacy parameter (epsilon) in differential privacy.
        delta (float): The privacy parameter (delta) in differential privacy.

    Returns:
        numpy.ndarray: The matrix with added Gaussian noise.
    """

    # Calculate the scale of the Gaussian noise
    scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    # Generate Gaussian noise
    noise = np.random.normal(loc=0, scale=scale, size=matrix.shape)

    # Add Gaussian noise to the matrix
    matrix_with_noise = matrix + noise

    return matrix_with_noise
    