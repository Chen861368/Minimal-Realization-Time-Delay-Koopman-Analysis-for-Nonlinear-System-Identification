# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm
import os

def visualize_matrix(A, title, save_path=None):
    """
    Visualizes a matrix as a heatmap with stylistic adjustments for enhanced readability and aesthetics.
    Optionally saves the plot as a PDF if save_path is provided.

    Parameters:
    - A: The matrix to visualize (2D numpy array)
    - title: The title of the plot (used for the saved file name)
    - save_path: The directory path where the plot will be saved (optional)
    """
    # Set the color range based on the maximum absolute value of the matrix
    vmax = np.abs(A).max()

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    # Create a meshgrid for the plot
    X, Y = np.meshgrid(np.arange(A.shape[1]+1), np.arange(A.shape[0]+1))
    
    # Invert the y-axis to match the standard matrix display
    ax.invert_yaxis()

    # Plot the matrix using pcolormesh with a color map
    pos = ax.pcolormesh(X, Y, A.real, cmap="seismic", vmax=vmax, vmin=-vmax)

    # Add color bar
    cbar = fig.colorbar(pos, ax=ax)
    cbar.ax.tick_params(labelsize=10)

    # Adjust the layout to avoid clipping
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # If a save_path is provided, save the plot as a PDF
    if save_path:
        filename = f"{save_path}/{title}.pdf"
        plt.savefig(filename, format='pdf', bbox_inches='tight')

    # Display the figure
    plt.show()



def create_hankel_matrix(data, rows, cols):
    """
    Create a Hankel matrix from the given data.

    Parameters:
    data (numpy.ndarray): The input data matrix with shape (n_samples, n_features).
    rows (int): The number of rows in the Hankel matrix.
    cols (int): The number of columns in the Hankel matrix.

    Returns:
    hankel_matrix (numpy.ndarray): The resulting Hankel matrix.
    """
    n_samples, n_features = data.shape
    hankel_matrix = np.zeros((cols, rows * n_features))

    for i in range(cols):
        for j in range(rows):
            if i + j < n_samples:
                hankel_matrix[i, j * n_features: (j + 1) * n_features] = data[i + j]

    return hankel_matrix.T


def svd(hankel_matrix):
    """
    Perform SVD on the Hankel matrix.

    Parameters:
    - hankel_matrix: Numpy array of the Hankel matrix.

    Returns:
    - U: Left singular vectors.
    - S: Singular values.
    - V: Right singular vectors.
    """
    U, S, Vt = np.linalg.svd(hankel_matrix, full_matrices=False)
    return U, S, Vt




def compute_dmd_and_c(U, S, V, n):
    """
    Compute the matrix C = U * S and the Dynamic Mode Decomposition (DMD) matrix A.

    Parameters:
    - U: numpy.ndarray, the left singular vectors (matrix).
    - S: numpy.ndarray, the singular values (vector).
    - V: numpy.ndarray, the right singular vectors (matrix).
    - n: int, the number of rows to return from the resulting matrix C.

    Returns:
    - A: numpy.ndarray, the approximated system matrix describing the dynamics.
    - C: numpy.ndarray, the resulting matrix C of shape (n, U.shape[1]).
    """
    # Part 1: Compute C = U * S
    # Convert S to a diagonal matrix
    S_diag = np.diag(S)

    # Compute C = U * S
    C = np.dot(U, S_diag)

    # Return the first n rows of C
    C_1 = C[:n, :]

    # Part 2: Compute the DMD matrix A
    # Split the displacement data into X and Y matrices using V
    # V.T because the original compute_dmd_matrix expects V transposed
    X = V.T[:, :-1]
    Y = V.T[:, 1:]

    # Compute the DMD matrix A
    A = Y @ X.T

    return A, C_1


def find_95_percent_threshold(singular_values):
    """
    Find the threshold at which the cumulative sum of singular values reaches 95% of the total sum.

    Parameters:
    - singular_values: Array of singular values.

    Returns:
    - index_95_percent: Index at which cumulative sum of singular values reaches 95%.
    """
    total_sum = np.sum(singular_values)
    cumulative_sum = np.cumsum(singular_values)
    index_95_percent = np.where(cumulative_sum >= 0.95 * total_sum)[0][0] + 1  # +1 to get the count

    return index_95_percent



def autonomous_state_space_from_hankel(y_data, rows_y, cols_y, model_order_y=None):
    """
    Create state-space representation from a Hankel matrix.

    Parameters:
    - y_data: Numpy array of output data (shape: n_samples, n_features).
    - rows_y: Number of rows in the Hankel matrix.
    - cols_y: Number of columns in the Hankel matrix.
    - model_order_y: Order of the state-space model (optional).

    Returns:
    - A: State matrix of the system.
    - C_y: Output matrix of the system.
    """
    hankel_matrix_y = create_hankel_matrix(y_data, rows_y, cols_y)
    
    m = y_data.shape[1]
    
    U, S, V = svd(hankel_matrix_y)
    
    # Determine the order based on model_order_y or 95% threshold
    if model_order_y is not None:
        index_to_keep = model_order_y
    else:
        index_to_keep = find_95_percent_threshold(S)
        print(f"The 95% of the singular values are captured by the first {index_to_keep} singular values.")
    
    # Keep the specified number of singular values and vectors
    U = U[:, :index_to_keep]
    S = S[:index_to_keep]
    V = V[:index_to_keep, :]
    
    A, C = compute_dmd_and_c(U, S, V.T, m)
    
    return A, C

def plot_eigenvalues(eigenvalues, save_path=None):
    """
    Plot eigenvalues on the complex plane with a unit circle, DPI=300, and save as PDF.

    Parameters:
    - eigenvalues: A list or array of eigenvalues (complex numbers).
    - save_path: The file path where the PDF will be saved (if None, the plot will not be saved).
    """
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Draw the unit circle (circle with radius 1)
    unit_circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--', linewidth=1.5, label='Unit Circle')
    ax.add_artist(unit_circle)
    
    # Plot eigenvalues as a scatter plot (real vs imaginary parts)
    scatter = ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), color='blue', s=20, zorder=5, 
                         edgecolor='black', linewidth=1, label='Eigenvalues')

    # Set axis limits and aspect ratio to make the plot look like a square
    ax.set_xlim([-1.8, 1.8])
    ax.set_ylim([-1.5, 1.5])
    ax.set_aspect('equal', 'box')

    # Add gridlines to the plot
    ax.grid(True, linestyle='--', alpha=0.6)

    # Add axis labels and title, using 'Times New Roman' font
    ax.set_xlabel('Real Part', fontdict={'family': 'Times New Roman', 'size': 18})
    ax.set_ylabel('Imaginary Part', fontdict={'family': 'Times New Roman', 'size': 18})

    # Beautify the legend, particularly focusing on eigenvalues
    legend = ax.legend(loc='upper right', fontsize=10)
    legend.legend_handles[1]._sizes = [100]  # Adjust the legend size for eigenvalues

    # If save_path is provided, save the plot as a PDF
    if save_path:
        filename = os.path.join(save_path, 'eigenvalues_plot.pdf')
        plt.savefig(filename, format='pdf', bbox_inches='tight')  # Ensures tight bounding box around the plot
    
    # Display the plot
    plt.show()
    
def save_matrices(A, C, prefix='dmd'):
    """
    Save matrices A and C to local files.

    Parameters:
    - A: numpy.ndarray, the DMD matrix.
    - C: numpy.ndarray, the matrix resulting from U * S.
    - prefix: A string prefix for the file names.
    """
    np.save(f'{prefix}_A_DMD.npy', A)
    np.save(f'{prefix}_C.npy', C)
    print(f'Matrices saved as {prefix}_A_DMD.npy and {prefix}_C.npy')

def discrete_to_continuous(A_d, sampling_frequency):
    """
    Converts a discrete matrix A_d to a continuous matrix A_c.

    Parameters:
    - A_d: np.array
        The discrete matrix to be converted.
    - sampling_frequency: float
        The sampling frequency used in the discrete system.

    Returns:
    - A_c: np.array
        The resulting continuous matrix.
    """
    # Calculate the time step Δt (inverse of sampling frequency)
    delta_t = 1 / sampling_frequency

    # Compute the continuous matrix A_c using the logarithm of A_d divided by Δt
    A_c = logm(A_d) / delta_t

    return A_c





# Load Lorenz data saved as a numpy array
load_path = 'Lorenz_data.npy'  # Assuming the data file is in the current working directory
X = np.load(load_path)  # Load the data from the specified file


Lorenz_data = X[:75000, :].T

rows = 30     # Number of rows in Lorenz_data
cols = 30000  # Number of columns in Lorenz_data (adjust this if needed for memory efficiency)

# Define the model order for the state-space model
model_order_y = 20  # Example value for the model order (this defines the number of states in the model)

# Compute the autonomous state-space matrices A and C_y from the Hankel matrix
A, C = autonomous_state_space_from_hankel(Lorenz_data, rows, cols, model_order_y)

# Visualize the matrix A (Lorenz system)
visualize_matrix(A, 'Lorenz')

# Convert the discrete matrix A to a continuous matrix A_c
A_c = discrete_to_continuous(A, 1000)

# Visualize the continuous matrix A_c
visualize_matrix(A_c, 'Lorenz_conti')

# Plot the eigenvalues of matrix A
plot_eigenvalues(np.linalg.eigvals(A))


save_matrices(A, C, prefix='Lorenz')










