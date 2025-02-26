# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
np.random.seed(0)
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.signal import welch


def evaluate_predictions(actual, predicted):
    """
    Evaluate the performance of predictions against actual values.

    Parameters:
    actual (array-like): The ground truth values.
    predicted (array-like): The predicted values.

    Returns:
    tuple: A tuple containing:
        - MSE (Mean Squared Error)
        - NMSE (Normalized Mean Squared Error)
        - R² (Coefficient of Determination)
    """
    # Convert inputs to numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Calculate MSE
    mse = np.mean((actual - predicted) ** 2)
    
    # Calculate NMSE
    nmse = mse / np.mean((actual - np.mean(actual)) ** 2)
    
    # Calculate R²
    ss_total = np.sum((actual - np.mean(actual)) ** 2)
    ss_residual = np.sum((actual - predicted) ** 2)
    r_squared = 1 - (ss_residual / ss_total)

    return mse, nmse, r_squared

def compare_psd(y, y_est, fs=10.0, save_path=None):
    """
    Compare the Power Spectral Density (PSD) of original data and estimated data with enhanced readability and aesthetics.

    Parameters:
    y (numpy.ndarray): The matrix of original responses (outputs) with shape (time_steps, n_outputs).
    y_est (numpy.ndarray): The matrix of estimated responses (outputs) with shape (time_steps, n_outputs).
    fs (float): The sampling frequency of the data.
    save_path (str): The path where the generated plots will be saved. If not provided, the plot will not be saved.

    Returns:
    None
    """
    # Set the style and context
    sns.set(style="whitegrid", context="talk")
    
    # Set global font to Times New Roman and increase size
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    
    n_outputs = y.shape[1]
    
    for i in range(n_outputs):
        fig, ax = plt.figure(figsize=(12, 8), dpi=300), plt.gca()
        
        # Compute PSD for original data
        f, Pxx_orig = welch(y[:, i], fs, nperseg=1024)
        ax.semilogy(f, Pxx_orig, label=f'True data', color='b', linewidth=1.5)
        
        # Compute PSD for estimated data
        f, Pxx_est = welch(y_est[:, i], fs, nperseg=1024)
        ax.semilogy(f, Pxx_est, label=f'Predicted data', color='r', linestyle='--', linewidth=1.5)
        
        plt.xlabel('Frequency (Hz)', fontsize=25)
        plt.ylabel('Power Spectral Density', fontsize=25)
        
        plt.legend(fontsize='x-large', handlelength=2, loc='upper right', frameon=True)
        plt.grid(True, linestyle='--')
        
        # Set tick parameters
        plt.tick_params(axis='both', which='major', labelsize=20)
        
        # Change the border color to black
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        
        # Only save the plot if save_path is provided
        if save_path:
            filename = f"{save_path}\\true_vs_actual_Lorenz_psd_{i+1}.pdf"
            plt.tight_layout()
            plt.savefig(filename, format='pdf', bbox_inches='tight')
        
        # Show the plot
        plt.show()

        
def visualize_results_monit(time, actual_data, estimated_data, n, iterations,save_path=None):
    """
    Visualize the true and estimated states with stylistic adjustments for enhanced readability
    and aesthetics, and save the plot as a PDF if a save path is provided.

    Parameters:
    - time: Time steps
    - actual_data: True displacements
    - estimated_data: Estimated states
    - n: Number of degrees of freedom
    - iterations: Number of iterations
    - save_path: The directory path where the plot will be saved
    """
    sns.set(style="white", context="talk")  # Use a white background and context suitable for presentations

    for i in range(n):
        plt.figure(figsize=(12, 8), dpi=300)  # Set figure size and DPI

        # Set global font to Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 16

        # Plot actual data with a solid black line
        plt.plot(time[:iterations], actual_data[i, :iterations], label=f'True data', linestyle='-', marker='', color='blue', linewidth=2.5)

        # Plot estimated data with a dashed red line
        plt.plot(time[:iterations], estimated_data[i, :iterations], label=f'Predicted data', linestyle=':', marker='', color='orange', linewidth=2.5)

        # Customize labels with explicit font sizes
        plt.xlabel('Time (t)', fontsize=25)
        plt.ylabel('y (t)', fontsize=25)

        # Enlarge legend line and text, add a black edge to the legend with a white background for visibility
        legend = plt.legend(fontsize='x-large', handlelength=2, edgecolor='black', loc='upper right', frameon=True, fancybox=False)
        # Set the linewidth of the legend border
        legend.get_frame().set_linewidth(1.5)

        # Explicitly set tick label sizes
        plt.tick_params(axis='both', which='major', labelsize=20)

        # Use dashed grid lines for better readability
        plt.grid(True, linestyle='--', linewidth=0.5)

        plt.tight_layout()

        # Only save the plot if save_path is provided
        if save_path:
            filename = f"{save_path}/true_vs_actual_Lorenz_{i+1}.pdf"
            plt.savefig(filename, format='pdf', bbox_inches='tight')

        plt.show()




def kalman_filter_update(A, C, Q, R, x_hat, P, y):
    """
    Perform the Kalman filter update step, which consists of two main parts: 
    prediction and correction.

    Parameters:
    - A: State transition matrix
    - C: Measurement matrix
    - Q: Process noise covariance matrix
    - R: Measurement noise covariance matrix
    - x_hat: Previous state estimate
    - P: Previous state covariance matrix
    - y: Measurement vector

    Returns:
    - x_hat: Updated state estimate
    - P: Updated state covariance matrix
    - C @ x_hat_pred: Kalman filter innovation (residual)
    """

    def predict(x_hat, P, A, Q):
        """
        Prediction step of the Kalman filter.

        - x_hat_pred: Predicted state estimate
        - P_pred: Predicted state covariance
        """
        x_hat_pred = A @ x_hat  # Predicted state estimate
        P_pred = A @ P @ A.T + Q  # Predicted state covariance
        return x_hat_pred, P_pred

    def update(x_hat_pred, P_pred, y, C, R):
        """
        Correction step of the Kalman filter.

        - S: Kalman gain innovation
        - K: Kalman gain
        """
        S = C @ P_pred @ C.T + R  # Innovation (measurement residual) covariance
        K = P_pred @ C.T @ np.linalg.inv(S)  # Kalman gain
        x_hat = x_hat_pred + K @ (y - C @ x_hat_pred)  # Corrected state estimate
        P = (np.eye(len(P_pred)) - K @ C) @ P_pred  # Updated state covariance matrix
        return x_hat, P

    # Prediction step
    x_hat_pred, P_pred = predict(x_hat, P, A, Q)

    # Update step
    x_hat, P = update(x_hat_pred, P_pred, y, C, R)
    
    return x_hat, P, C @ x_hat_pred  # Return the updated state estimate, covariance, and innovation



def APSM_algorithm(combined_matrix, learning_rate, iterations, A, C_matrix, P, Q, R, S):
    """
    Implements the APSM (Adaptive Physics-Informed System Modeling) algorithm for state estimation.

    Parameters:
    - combined_matrix: Combined input-output data (e.g., system inputs and outputs).
    - learning_rate: Learning rate used for gradient descent updates.
    - iterations: Number of iterations for the algorithm to run.
    - A: Initial state transition matrix.
    - C_matrix: Observation matrix (relates the state to the measurement).
    - P: Initial error covariance matrix (represents uncertainty in the state estimate).
    - Q: Process noise covariance matrix (uncertainty in the process model).
    - R: Measurement noise covariance matrix (uncertainty in the measurement model).
    - S: Cholesky factor of the process noise covariance matrix (for numerical stability).
    
    Returns:
    - x_estimates: Estimated state values for each iteration.
    - A: Updated state transition matrix after the algorithm.
    - y_estimates: Estimated measurements corresponding to each iteration.
    """
    n = A.shape[0]  # Number of states
    m = C_matrix.shape[0]  # Number of measurements
    x_hat = np.zeros((n, 1))  # Initial state estimate
    x_estimates = np.zeros((n, iterations))  # Array to store state estimates for each iteration
    y_estimates = np.zeros((m, iterations))  # Array to store measurement estimates for each iteration

    for k in range(iterations):
        x_hat_old = x_hat  # Store the old state estimate
        y_k = combined_matrix[:, k].reshape(-1, 1)  # Get the current measurement
        
        # Perform Kalman filter update with the current state transition matrix A
        x_hat, P, y_pred = kalman_filter_update(A, C_matrix, Q, R, x_hat_old, P, y_k)

        # Update the state transition matrix A using gradient descent (commented out for now)
        # A = gradient_descent_update(A, x_hat_old, x_hat, learning_rate)
        
        # Save the new state estimate into the estimates array
        x_estimates[:, k] = x_hat.flatten()
        y_estimates[:, k] = y_pred.flatten()

    return x_estimates, A, y_estimates




def plot_residuals(time, actual_data, estimated_data, save_path=None):
    """
    Visualize the residuals (the difference between actual and estimated data) with stylistic adjustments 
    for enhanced readability and aesthetics, and save the plot as a PDF if a save path is provided.

    Parameters:
    - time: Time steps
    - actual_data: True values
    - estimated_data: Estimated values
    - save_path: The directory path where the plot will be saved (optional)
    """
    # Calculate residuals
    residuals = actual_data - estimated_data

    n = residuals.shape[0]  # Get the number of data dimensions
    for i in range(n):
        plt.figure(figsize=(12, 8), dpi=300)  # Create the figure

        # Set global font to Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 16

        # Plot the residuals
        plt.plot(time, residuals[i, :], label=f'Residual {i+1}', linestyle='-', color='blue', linewidth=2.5)

        # Set labels
        plt.xlabel('Time (t)', fontsize=25)
        plt.ylabel('Residual', fontsize=25)

        # Add legend
        plt.legend(fontsize='x-large', loc='upper right')

        # Set grid and other styling
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()

        # Only save the plot if save_path is provided
        if save_path:
            filename = f"{save_path}/residual_plot_{i+1}.pdf"
            plt.savefig(filename, format='pdf', bbox_inches='tight')

        # Display the plot
        plt.show()


def plot_residual_histograms(actual_data, estimated_data, save_path=None):
    """
    Visualize the residuals' histograms (difference between actual and estimated data) with stylistic adjustments 
    for enhanced readability and aesthetics, and save the plot as a PDF if a save path is provided.

    Parameters:
    - actual_data: True values
    - estimated_data: Estimated values
    - save_path: The directory path where the plot will be saved (optional)
    """
    # Calculate residuals
    residuals = actual_data - estimated_data

    n = residuals.shape[0]  # Get the number of coordinates (e.g., x, y, z)
    for i in range(n):
        plt.figure(figsize=(10, 6), dpi=300)  # Create figure object

        # Set global font to Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 16

        # Plot the histogram of residuals
        plt.hist(residuals[i, :], bins=50, color='blue', alpha=0.7, label=f'Residual {i+1}')

        # Set labels
        plt.xlabel('Residual Value', fontsize=20)
        plt.ylabel('Frequency', fontsize=20)

        # Add legend
        plt.legend(fontsize='x-large', loc='upper right')

        # Set grid and other plot styles
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()

        # Save the plot as a PDF only if save_path is provided
        if save_path:
            filename = f"{save_path}/residual_histogram_{i+1}.pdf"
            plt.savefig(filename, format='pdf', bbox_inches='tight')

        # Display the plot
        plt.show()

def load_matrices(prefix='dmd'):
    """
    Load matrices A and C from local files.

    Parameters:
    - prefix: A string prefix for the file names.

    Returns:
    - A: numpy.ndarray, the DMD matrix.
    - C: numpy.ndarray, the matrix resulting from U * S.
    """
    A = np.load(f'{prefix}_A_DMD.npy')
    C = np.load(f'{prefix}_C.npy')
    return A, C

def compare_phase_space(actual_data, estimated_data, xlabel="x", ylabel="z", save_path=None):
    """
    Compare the 2D phase space (x-z) between actual and predicted data, 
    and plot them together with proper labeling and grid style for visual clarity.

    Parameters:
    - actual_data: Actual data array with shape (2, N), where rows represent x and z.
    - estimated_data: Predicted data array with shape (2, N), where rows represent x and z.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - save_path: (Optional) If provided, the plot will be saved as a PDF at this path.
    """
    # Extract the x and z dimensions from the actual and estimated data
    x_true = actual_data[0, :]
    z_true = actual_data[1, :]
    x_pred = estimated_data[0, :]
    z_pred = estimated_data[1, :]

    # Set global font to 'Times New Roman'
    plt.rcParams["font.family"] = "Times New Roman"

    # Create the figure object with size and DPI for clarity
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # Plot the true data (x vs z) in blue
    ax.plot(x_true, z_true, color='blue', lw=1.5, label="True Data")

    # Plot the predicted data (x vs z) in red with dashed lines
    ax.plot(x_pred, z_pred, color='orange', linestyle='--', lw=1.5, label="Predicted Data")

    # Set the axis labels
    ax.set_xlabel(xlabel, labelpad=15, fontsize=20, fontname='Times New Roman')
    ax.set_ylabel(ylabel, labelpad=15, fontsize=20, fontname='Times New Roman', rotation=0)

    # Adjust the y-axis label position for clarity
    ax.yaxis.set_label_coords(-0.05, 0.5)

    # Beautify the tick labels with 'Times New Roman'
    ax.tick_params(axis='both', which='major', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    # Add grid lines in grey dashed style for better readability
    ax.grid(True, linestyle='--', linewidth=0.75, color='gray', alpha=0.7)

    # Add a legend in the upper right corner with a border
    legend = ax.legend(fontsize=20, loc='upper right')
    legend.get_frame().set_edgecolor('black')

    # Customize the border thickness for a cleaner look
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Ensure the layout is tight and does not overflow the plot area
    plt.tight_layout()

    # Save the plot to a file if a save path is provided
    if save_path:
        filename = f"{save_path}/Phase_space_comparison.pdf"
        plt.savefig(filename, format='pdf', bbox_inches='tight')

    # Display the plot
    plt.show()



def plot_residual_phase_space(actual_data, estimated_data, xlabel="x", ylabel="z", save_path=None):
    """
    Plot the residual (difference) between actual and predicted data in the x-z phase space,
    and use color to represent the magnitude of the residuals. This enables clear correspondence
    between x and z while the residual values are represented through color intensity.

    Parameters:
    - actual_data: Actual data array with shape (2, N), where the first row represents x and the second row represents z.
    - estimated_data: Predicted data array with shape (2, N), where the first row represents x and the second row represents z.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - save_path: (Optional) If provided, the plot will be saved as a PDF at this path.
    """
    # Compute the residuals (difference between actual and predicted)
    x_residual = actual_data[0, :] - estimated_data[0, :]
    z_residual = actual_data[1, :] - estimated_data[1, :]
    
    # Calculate the magnitude of the residuals (for color representation)
    residual_magnitude = np.sqrt(x_residual**2 + z_residual**2)
    actual_magnitude = np.sqrt(actual_data[0, :]**2 + actual_data[1, :]**2)
    
    # Avoid division by zero by adding a small epsilon
    # epsilon = 1e-8
    residual_magnitude = residual_magnitude / (actual_magnitude)

    # Set global font to 'Times New Roman'
    plt.rcParams["font.family"] = "Times New Roman"

    # Create the figure object with size and DPI for clarity
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Use scatter plot to plot x vs z with the color based on the residual magnitude
    scatter = ax.scatter(actual_data[0, :], actual_data[1, :], c=residual_magnitude, cmap='viridis', s=1, label="Residual Magnitude")

    # Set the axis labels with a closer x-label (reduced labelpad)
    ax.set_xlabel(xlabel, labelpad=5, fontsize=20, fontname='Times New Roman')  # Reduced labelpad for x-axis
    ax.set_ylabel(ylabel, labelpad=15, fontsize=20, fontname='Times New Roman', rotation=0)

    # Adjust the y-axis label position for clarity
    ax.yaxis.set_label_coords(-0.07, 0.5)

    # Beautify the tick labels with 'Times New Roman'
    ax.tick_params(axis='both', which='major', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    # Add grid lines in grey dashed style for better readability
    ax.grid(True, linestyle='--', linewidth=0.75, color='gray', alpha=0.7)

    # Add a color bar to represent the residual magnitude
    cbar = fig.colorbar(scatter, ax=ax)

    # Customize the border thickness for a cleaner look
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Ensure the layout is tight and does not overflow the plot area
    plt.tight_layout()

    # Save the plot to a file if a save path is provided
    if save_path:
        filename = f"{save_path}/Residual_phase_space_with_color.pdf"
        plt.savefig(filename, format='pdf', bbox_inches='tight')

    # Display the plot
    plt.show()


# Plotting function
def plot_coordinate_comparison(X_actual, X_predicted, coordinate_labels=None):
    """
    Plot comparison for each coordinate in the actual and predicted data.
    
    Parameters:
    - X_actual: numpy.ndarray, actual data matrix with shape (n, t).
    - X_predicted: numpy.ndarray, predicted data matrix with shape (n, t).
    - coordinate_labels: list of str, optional, labels for each coordinate.
    """
    num_coordinates = X_actual.shape[0]
    
    plt.figure(figsize=(10, num_coordinates * 3), dpi=300)
    
    for i in range(num_coordinates):
        plt.subplot(num_coordinates, 1, i + 1)
        plt.plot(X_actual[i, :], label='Actual', color='blue')
        plt.plot(X_predicted[i, :], label='Predicted', linestyle='--', color='red')
        plt.ylabel(coordinate_labels[i] if coordinate_labels else f"Coordinate {i+1}")
        plt.xlabel("Time Step")
        plt.legend()
        plt.title(f"Comparison for {coordinate_labels[i] if coordinate_labels else 'Coordinate'} {i+1}")
    
    plt.tight_layout()
    plt.show()
    
    
# ============ Parameters ============

# Load Lorenz data saved as a numpy array
load_path = 'Lorenz_data.npy'  # Assuming the data file is in the current working directory
X = np.load(load_path)  # Load the data from the specified file

# Extract a portion of the data for further processing
y = X[:,75000:].T  # Time series data from index 75000 onwards

# Define the time steps for the extracted data
time = np.linspace(75, 150, y.shape[0])

# Load the DMD matrices A and C
A_DMD, C_matrix = load_matrices(prefix='Lorenz')

# Get dimensions of matrices A and C
n = A_DMD.shape[1]  # Number of states
m = C_matrix.shape[0]  # Number of measurements

# Initialize parameters
B = np.zeros((n, 1))  # Control input matrix (initialized as zero)
# C_matrix = np.eye(n)  # Output matrix (assuming complete observation)
P = np.eye(n) * 0.01  # Initial error covariance matrix with small values
Q = np.eye(n) * 0.01  # Process noise covariance matrix
R = np.eye(m) * 0.00001  # Measurement noise covariance matrix
S = np.linalg.cholesky(np.eye(n) * 1e-3)  # Cholesky factor of process noise covariance
learning_rate = 0  # Learning rate for updates (set to 0 here)
iterations = y.shape[0]  # Number of iterations (equal to the length of the data)

# Run the APSM (Adaptive Prediction State Model) algorithm
x_estimates, A, y_estimates = APSM_algorithm(y.T, learning_rate, iterations, A_DMD, C_matrix, P, Q, R, S)

# Evaluate the predictions, skipping the first time step
y_mse, y_nmse, y_r2 = evaluate_predictions(y.T[:, 1:], y_estimates[:, 1:])

# Print evaluation metrics (Normalized Mean Squared Error)
print(y_nmse)

# Visualize the results of monitoring the true vs estimated data
visualize_results_monit(time[1:], y.T[:, 1:], y_estimates[:, 1:], 3, iterations - 1)

# Compare Power Spectral Density (PSD), skipping the first time step
compare_psd(y[1:, :], y_estimates[:, 1:].T)

# Call function to plot residuals
plot_residuals(time[1:], y.T[:, 1:], y_estimates[:, 1:])

# Call function to plot residual histograms
plot_residual_histograms(y.T[:, 1:], y_estimates[:, 1:])

# Example usage: Compare phase space between true and estimated data
compare_phase_space(y.T[:, 1:], y_estimates[:, 1:])

# Example usage: Plot residual phase space between true and estimated data
plot_residual_phase_space(y.T[:, 1:], y_estimates[:, 1:])





