import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from varying_antennas import *
from varying_Pn import *

# Set font family globally
plt.rcParams['font.family'] = 'Times New Roman'

def calculate_average(data_list):
    """Calculate the average of a list of values"""
    return np.mean(data_list)

def get_marker(algorithm_name):
    """Get the marker symbol for a given algorithm"""
    markers = {
        'MADDPG': '^',
        'MATD3': 's',
        'MASAC': 'v',
        'MAPPO': 'D',
        'Greedy': 'o',
        'Random': 'x'
    }
    return markers.get(algorithm_name, 'o')

def get_color(algorithm_name):
    """Get the color for a given algorithm"""
    colors = {
        'MADDPG': 'blue',
        'MATD3': 'green',
        'MASAC': 'lightblue',
        'MAPPO': 'red',
        'Greedy': 'orange',
        'Random': 'black'
    }
    return colors.get(algorithm_name, 'blue')

def plot_energy_efficiency_vs_transmit_power():
    """Plot (a): Average Energy Efficiency vs Transmit Power"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Transmit power values (dBm) - converting from normalized values
    transmit_powers = [27, 30, 33]  # dBm
    
    # Calculate averages for each algorithm and power level
    algorithms = {
        'MADDPG': {
            'Pn_05': maddpg_05,
            'Pn_1': maddpg_1,
            'Pn_15': maddpg_15
        },
        'MATD3': {
            'Pn_05': matd3_05,
            'Pn_1': matd3_1,
            'Pn_15': matd3_15
        },
        'MASAC': {
            'Pn_05': masac_05,
            'Pn_1': masac_1,
            'Pn_15': masac_15
        },
        'MAPPO': {
            'Pn_05': mappo_05,
            'Pn_1': mappo_1,
            'Pn_15': mappo_15
        },
        'Greedy': {
            'Pn_05': greedy_05,
            'Pn_1': greedy_1,
            'Pn_15': greedy_15
        },
        'Random': {
            'Pn_05': random_05,
            'Pn_1': random_1,
            'Pn_15': random_15
        }
    }
    
    # Plot each algorithm
    for alg_name, alg_data in algorithms.items():
        averages = []
        for power_key in ['Pn_05', 'Pn_1', 'Pn_15']:
            avg = calculate_average(alg_data[power_key])
            averages.append(avg)
        
        # Create smooth curve with cubic spline interpolation
        if len(transmit_powers) > 1:
            cs = CubicSpline(transmit_powers, averages, bc_type='natural')
            x_smooth = np.linspace(min(transmit_powers), max(transmit_powers), 100)
            y_smooth = cs(x_smooth)
            
            # Plot smooth curve with evenly spaced markers using 'markevery'
            ax.plot(x_smooth, y_smooth, 
                    label=alg_name, 
                    color=get_color(alg_name),
                    marker=get_marker(alg_name),
                    markersize=8,
                    markevery=20,  # <-- Added/Updated: Markers every 20 points
                    linewidth=2)
            
        else:
            # Fallback for single data point
            ax.plot(transmit_powers, averages, label=alg_name, marker=get_marker(alg_name), markersize=6, color=get_color(alg_name))
    
    # Customize the plot
    ax.set_xlabel('Transmit Power (dBm)', fontsize=26)
    ax.set_ylabel('Average Energy Efficiency (b/J)', fontsize=26)
    
    # Create custom legend with markers (Line2D objects handle line and marker styles)
    legend_elements = []
    for alg_name in algorithms.keys():
        # Ensure the legend element uses the marker
        legend_elements.append(plt.Line2D([0], [0], color=get_color(alg_name), 
                                         marker=get_marker(alg_name), markersize=8, # Use the larger markersize
                                         label=alg_name, linewidth=2))
    ax.legend(handles=legend_elements, fontsize=18, markerfirst=True)
    
    # Set x-axis limits to remove extra space
    ax.set_xlim(27, 33)
    
    # Set x-axis ticks to show only the three data points
    ax.set_xticks(transmit_powers)
    ax.tick_params(axis='both', which='major', labelsize=22)
    
    # Add grid with lightgray color and thin lines
    ax.grid(True, linestyle=':', color='lightgray', linewidth=0.2)
    
    return fig

def plot_energy_efficiency_vs_antennas():
    """Plot (b): Average Energy Efficiency vs Number of RF-EH antennas"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Number of RF-EH antennas
    antenna_numbers = [1, 2, 3]
    
    # Calculate averages for each algorithm and antenna number
    algorithms = {
        'MADDPG': {
            'L_1': maddpg_1,
            'L_2': maddpg_2,
            'L_3': maddpg_3
        },
        'MATD3': {
            'L_1': matd3_1,
            'L_2': matd3_2,
            'L_3': matd3_3
        },
        'MASAC': {
            'L_1': masac_1,
            'L_2': masac_2,
            'L_3': masac_3
        },
        'MAPPO': {
            'L_1': mappo_1,
            'L_2': mappo_2,
            'L_3': mappo_3
        },
        'Greedy': {
            'L_1': greedy_1,
            'L_2': greedy_2,
            'L_3': greedy_3
        },
        'Random': {
            'L_1': random_1,
            'L_2': random_2,
            'L_3': random_3
        }
    }
    
    # Plot each algorithm
    for alg_name, alg_data in algorithms.items():
        averages = []
        for antenna_key in ['L_1', 'L_2', 'L_3']:
            avg = calculate_average(alg_data[antenna_key])
            averages.append(avg)
        
        # Create smooth curve with cubic spline interpolation
        if len(antenna_numbers) > 1:
            cs = CubicSpline(antenna_numbers, averages, bc_type='natural')
            x_smooth = np.linspace(min(antenna_numbers), max(antenna_numbers), 100)
            y_smooth = cs(x_smooth)
            
            # Plot smooth curve with evenly spaced markers using 'markevery'
            ax.plot(x_smooth, y_smooth, 
                    label=alg_name, 
                    color=get_color(alg_name),
                    marker=get_marker(alg_name),
                    markersize=8,
                    markevery=20,  # <-- Added/Updated: Markers every 20 points
                    linewidth=2)
            
        else:
            # Fallback for single data point
            ax.plot(antenna_numbers, averages, label=alg_name, marker=get_marker(alg_name), markersize=6, color=get_color(alg_name))
    
    # Customize the plot
    ax.set_xlabel('Number of RF-EH antennas on each SU', fontsize=26)
    ax.set_ylabel('Average Energy Efficiency (b/J)', fontsize=26)
    
    # Create custom legend with markers (Line2D objects handle line and marker styles)
    legend_elements = []
    for alg_name in algorithms.keys():
        # Ensure the legend element uses the marker
        legend_elements.append(plt.Line2D([0], [0], color=get_color(alg_name), 
                                         marker=get_marker(alg_name), markersize=8, # Use the larger markersize
                                         label=alg_name, linewidth=2))
    ax.legend(handles=legend_elements, loc='center left', fontsize=18, bbox_to_anchor=(0.01, 0.70), markerfirst=True)
    
    # Set x-axis limits to remove extra space
    ax.set_xlim(1, 3)
    
    # Set x-axis ticks to show only the three data points
    ax.set_xticks(antenna_numbers)
    ax.tick_params(axis='both', which='major', labelsize=22)
    
    # Add grid with lightgray color and thin lines
    ax.grid(True, linestyle=':', color='lightgray', linewidth=0.2)
    
    return fig

def main():
    """Create both plots and save them"""
    print("Creating energy efficiency vs transmit power plot...")
    fig_a = plot_energy_efficiency_vs_transmit_power()
    fig_a.savefig('energy_efficiency_vs_transmit_power.png', dpi=300, bbox_inches='tight')
    fig_a.savefig('energy_efficiency_vs_transmit_power.eps', format='eps', bbox_inches='tight')
    print("Saved: energy_efficiency_vs_transmit_power.png and .eps")
    plt.close(fig_a)
    
    print("Creating energy efficiency vs antennas plot...")
    fig_b = plot_energy_efficiency_vs_antennas()
    fig_b.savefig('energy_efficiency_vs_antennas.png', dpi=300, bbox_inches='tight')
    fig_b.savefig('energy_efficiency_vs_antennas.eps', format='eps', bbox_inches='tight')
    print("Saved: energy_efficiency_vs_antennas.png and .eps")
    plt.close(fig_b)
    
    print("\n" + "="*50)
    print("All plots have been created successfully!")
    print("="*50)
    print("Energy Efficiency vs Transmit Power plot:")
    print("- energy_efficiency_vs_transmit_power.png/.eps")
    print("\nEnergy Efficiency vs Antennas plot:")
    print("- energy_efficiency_vs_antennas.png/.eps")
    print("="*50)

if __name__ == "__main__":
    main()