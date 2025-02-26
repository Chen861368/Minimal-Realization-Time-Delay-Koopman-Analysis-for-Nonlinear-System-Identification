# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def plot_lorenz_attractor(X, save_path="C:\\Users\\HIT\\Desktop"):
    """
    Plot the 3D Lorenz attractor without title, using 'Times New Roman' font.
    
    :param X: A 3D data matrix of the Lorenz system (each column corresponds to one dimension of the attractor)
    :param save_path: (Optional) The file path to save the figure (default is the Desktop)
    """
    # Create a 3D plot with constrained layout for better arrangement
    fig1 = plt.figure(figsize=(10, 6), dpi=300, constrained_layout=True)
    ax1 = fig1.add_subplot(111, projection="3d")

    # Plot the 3D Lorenz attractor
    ax1.plot(X[0], X[1], X[2], lw=0.5, color='b')

    # Set axis labels with 'Times New Roman' font
    ax1.set_xlabel("x", labelpad=12, fontsize=12, color='black', fontname='Times New Roman')
    ax1.set_ylabel("y", labelpad=12, fontsize=12, color='black', fontname='Times New Roman')
    ax1.set_zlabel("z", labelpad=12, fontsize=12, color='black', fontname='Times New Roman')

    # Beautify tick labels with 'Times New Roman'
    ax1.tick_params(axis='both', which='major', labelsize=10, colors='black')
    for label in ax1.get_xticklabels() + ax1.get_yticklabels() + ax1.get_zticklabels():
        label.set_fontname('Times New Roman')

    # Remove background panes for a cleaner look
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.set_tick_params(width=1.5)
    ax1.yaxis.set_tick_params(width=1.5)
    ax1.zaxis.set_tick_params(width=1.5)

    # Add grid lines with adjusted transparency
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    # Set axis line colors
    ax1.xaxis.line.set_color((0.5, 0.5, 0.5, 0.8))
    ax1.yaxis.line.set_color((0.5, 0.5, 0.5, 0.8))
    ax1.zaxis.line.set_color((0.5, 0.5, 0.5, 0.8))
    plt.tight_layout()

    # Save as PDF if a save path is provided
    if save_path:
        filename = f"{save_path}/Lorenz_attractor_plot.pdf"
        plt.savefig(filename, format='pdf', bbox_inches='tight')

    # Display the plot
    plt.show()


def plot_time_series(t, x, save_path="C:\\Users\\HIT\\Desktop"):
    """
    Plot a time series graph with enhanced visual appeal, suitable for research reports.
    
    :param t: Time data (e.g., time steps or timestamps)
    :param x: x-dimension data from the Lorenz system
    :param save_path: (Optional) The file path to save the figure (default is the Desktop)
    """
    # Set global font to 'Times New Roman'
    plt.rcParams["font.family"] = "Times New Roman"

    # Create figure object, adjusting size and DPI for clarity and print suitability
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Plot the time series of x with blue color, thick line, and labeled as 'x(t)'
    ax.plot(t, x, color='blue', lw=1.5, label='x(t)')

    # Set axis labels with 'Times New Roman' font for a professional look
    ax.set_xlabel("Time (t)", labelpad=20, fontsize=20, fontname='Times New Roman')
    ax.set_ylabel("x", labelpad=15, fontsize=20, fontname='Times New Roman', rotation=0)

    # Adjust y-axis label position to avoid overlap with the axis
    ax.yaxis.set_label_coords(-0.05, 0.5)

    # Beautify tick labels with 'Times New Roman' font
    ax.tick_params(axis='both', which='major', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    # Add grid lines with gray dashed style for better contrast
    ax.grid(True, linestyle='--', linewidth=0.75, color='gray', alpha=0.7)

    # Set legend with black border for contrast and professional appearance
    legend = ax.legend(loc='upper right', fontsize=20, frameon=True)
    legend.get_frame().set_edgecolor('black')  # Set legend border color to black
    legend.get_frame().set_linewidth(1.2)  # Set legend border line width

    # Adjust plot border line width for clarity
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Ensure compact layout with tight layout
    plt.tight_layout()

    # If a save path is provided, save the plot as a high-quality PDF
    if save_path:
        filename = f"{save_path}/Time_series_plot.pdf"
        plt.savefig(filename, format='pdf', bbox_inches='tight')

    # Display the plot
    plt.show()




def generate_lorenz_data(t_eval):
    """
    Given a time vector t_eval = t1, t2, ..., evaluates and returns
    the snapshots of the Lorenz system as columns of the matrix X.
    """

    def lorenz_system(t, state):
        sigma, rho, beta = 10, 28, 8 / 3  # chaotic parameters
        x, y, z = state
        x_dot = sigma * (y - x)
        y_dot = (x * (rho - z)) - y
        z_dot = (x * y) - (beta * z)
        return [x_dot, y_dot, z_dot]

    # Set integrator keywords to replicate the odeint defaults
    integrator_keywords = {}
    integrator_keywords["rtol"] = 1e-12
    integrator_keywords["atol"] = 1e-12
    integrator_keywords["method"] = "LSODA"

    sol = solve_ivp(
        lorenz_system,
        [t_eval[0], t_eval[-1]],
        [-8, 8, 27],
        t_eval=t_eval,
        **integrator_keywords,
    )

    return sol.y


def get_ind_switch_lorenz(x):
    """
    Get indices of true lobe switching of the Lorenz system given x data.
    """
    ind_switch = np.sign(x[:-1]) - np.sign(x[1:]) != 0
    ind_switch = np.append(ind_switch, False)
    ind_switch = np.where(ind_switch)[0]

    return ind_switch

def plot_phase_space(x, y, xlabel="x (t)", ylabel="y (t)", save_path="C:\\Users\\HIT\\Desktop"):
    """
    Plot the x-y phase space diagram with enhanced visual appeal, suitable for research reports.
    
    :param x: x dimension data from the Lorenz system
    :param y: y dimension data from the Lorenz system
    :param xlabel: The label for the x-axis
    :param ylabel: The label for the y-axis
    :param save_path: (Optional) The file path to save the figure (default is the Desktop)
    """
    # Set global font to 'Times New Roman'
    plt.rcParams["font.family"] = "Times New Roman"

    # Create figure object, adjusting the size and DPI for clarity and print suitability
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Plot the x-y phase space curve with thicker blue lines for enhanced visual contrast
    ax.plot(x, y, color='blue', lw=1.5)

    # Set axis labels using the provided xlabel and ylabel
    ax.set_xlabel(xlabel, labelpad=15, fontsize=20, fontname='Times New Roman')
    ax.set_ylabel(ylabel, labelpad=15, fontsize=20, fontname='Times New Roman', rotation=0)

    # Adjust the y-axis label position to avoid overlap with the axis, ensuring clarity
    ax.yaxis.set_label_coords(-0.05, 0.5)

    # Beautify tick labels with 'Times New Roman' font
    ax.tick_params(axis='both', which='major', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    # Add grid lines with gray dashed style for better accuracy and contrast
    ax.grid(True, linestyle='--', linewidth=0.75, color='gray', alpha=0.7)

    # Set border line width for the plot to make it clearer
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Use tight_layout to ensure the layout is compact without overflowing elements
    plt.tight_layout()

    # Save the plot as a high-quality PDF if a save path is provided
    if save_path:
        filename = f"{save_path}/Phase_space_plot.pdf"
        plt.savefig(filename, format='pdf', bbox_inches='tight')

    # Display the plot
    plt.show()





# Generate Lorenz system data
dt = 0.001  # time step
m = 150000  # number of data samples
t = np.arange(m) * dt  # Time array from 0 to (m-1) * dt
X = generate_lorenz_data(t)  # Generate Lorenz system data (3-dimensional)
x = X[0]  # Extract x component (Lorenz system's x data)
# y = X[1]  # Extract y component (Lorenz system's y data)
# Uncomment above line if you need the y component

# Call the function to plot the 3D Lorenz attractor
plot_lorenz_attractor(X)  # Pass the generated data to plot 3D attractor

# Call the function to plot the time series of x
plot_time_series(t, x)  # Pass time and x data to plot the time series of x

# Call the function to plot the x-y phase space
plot_phase_space(X[0], X[1], xlabel="x", ylabel="y")  # Plot x vs y for phase space plot

# Uncomment the following lines to plot x-z and y-z phase space if needed
# plot_phase_space(X[0], X[2], xlabel="x", ylabel="z")  # Plot x vs z for phase space plot
# plot_phase_space(X[1], X[2], xlabel="y", ylabel="z")  # Plot y vs z for phase space plot

# # Set the path to save the generated Lorenz system data
# save_path = 'D:\\博士课题\\小论文\\environment load model\\论文代码\\Lorenz_data.npy'

# # Save the generated Lorenz system data as a numpy file
# np.save(save_path, X)  # Save the data to a .npy file for later use











