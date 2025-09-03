import matplotlib.pyplot as plt
import numpy as np
from plots_ee import *
from plots_sr_eh import *
from scipy.ndimage import gaussian_filter1d

# Set the style for better-looking plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.1

def smooth_data(data, sigma=1.0):
    """
    Apply Gaussian smoothing to the data
    """
    return gaussian_filter1d(data, sigma=sigma)

def plot_metric_comparison(env_name, env_data, title_suffix, metric_name, ylabel):
    """
    Plot metric comparison for a specific environment
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors, linestyles, and markers for each algorithm
    algorithms = {
        'MADDPG': {'color': 'blue', 'linestyle': '-', 'marker': '^'},
        'MATD3': {'color': 'green', 'linestyle': '-', 'marker': 's'},
        'MASAC': {'color': 'lightblue', 'linestyle': '-', 'marker': 'v'},
        'MAPPO': {'color': 'red', 'linestyle': '-', 'marker': 'D'},
        'Greedy': {'color': 'orange', 'linestyle': '-', 'marker': 'o'},
        'Random': {'color': 'black', 'linestyle': '-', 'marker': 'x'}
    }
    
    # Find the minimum length among all algorithms for this environment
    min_length = min(len(data) for data in env_data.values())
    print(f"Using minimum length {min_length} for {env_name} environment - {metric_name}")
    
    # Plot each algorithm
    for alg_name, alg_data in env_data.items():
        # Truncate to minimum length and smooth the data
        truncated_data = alg_data[:min_length]
        smoothed_data = smooth_data(truncated_data, sigma=1.5)
        
        color = algorithms[alg_name]['color']
        linestyle = algorithms[alg_name]['linestyle']
        marker = algorithms[alg_name]['marker']
        
        episodes = np.arange(1, min_length + 1)
        
        ax.plot(episodes, smoothed_data, color=color, 
                linestyle=linestyle, linewidth=2, marker=marker, 
                markersize=6, markevery=10, label=alg_name, alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Episodes', fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.grid(True, color='lightgray', linewidth=0.5)
    
    # Place legend outside the plot area to avoid covering data
    # First try to place it in the upper right outside
    legend = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                      fontsize=16, frameon=True, edgecolor="black")
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((1, 1, 1, 0))
    
    # If the legend would be cut off, try different positions
    fig = ax.get_figure()
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox_data = bbox.transformed(fig.transFigure.inverted())
    
    # If legend extends beyond figure bounds, move it to center right inside
    # For MRC, Simple, and SC energy efficiency plots, use top right
    # For sum rate plots, use lower right
    # For energy harvesting plots, use lower right with adjusted position
    if bbox_data.x1 > 0.95:
        legend.remove()
        if 'Energy Efficiency' in metric_name and env_name in ['MRC', 'Simple', 'SC']:
            legend = ax.legend(loc='upper right', fontsize=16, frameon=True, edgecolor="black")
            legend.get_frame().set_alpha(None)
            legend.get_frame().set_facecolor((1, 1, 1, 0))
        elif 'Sum Rate' in metric_name:
            legend = ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.4), fontsize=16, frameon=True, edgecolor="black")
            legend.get_frame().set_alpha(None)
            legend.get_frame().set_facecolor((1, 1, 1, 0))
        elif 'Energy Harvesting' in metric_name:
            legend = ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.3), fontsize=16, frameon=True, edgecolor="black")
            legend.get_frame().set_alpha(None)
            legend.get_frame().set_facecolor((1, 1, 1, 0))
        else:
            legend = ax.legend(loc='center right', fontsize=16, frameon=True, edgecolor="black")
            legend.get_frame().set_alpha(None)
            legend.get_frame().set_facecolor((1, 1, 1, 0))
    
    # Set x-axis limits and ticks
    ax.set_xlim(0, min_length)
    ax.set_xticks(np.arange(0, min_length + 1, min_length // 8))
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Set y-axis limits based on data range
    all_data = []
    for data in env_data.values():
        truncated = data[:min_length]
        smoothed = smooth_data(truncated, sigma=1.5)
        all_data.extend(smoothed)
    
    y_max = np.max(all_data) * 1.05
    y_min = np.min(all_data) * 0.95
    ax.set_ylim(y_min, y_max)
    
    # Use custom formatting for energy harvesting y-axis (show values in mJ without scientific notation)
    if 'Energy Harvesting' in metric_name:
        # Convert values to mJ (multiply by 1000) and format without scientific notation
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1000:.1f}'))
        # Update y-axis label to show mJ unit
        ax.set_ylabel('Energy Harvesting (mJ)', fontsize=18)
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Add zoomed-in inset for Greedy and Random algorithms (for all plot types except Sum Rate)
    if 'Greedy' in env_data and 'Random' in env_data and 'Sum Rate' not in metric_name:
        # Create inset axes for zoom
        axins = ax.inset_axes([0.35, 0.05, 0.15, 0.08])  # [x, y, width, height]
        
        # Plot Greedy and Random in the inset
        greedy_data = env_data['Greedy'][:min_length]
        random_data = env_data['Random'][:min_length]
        
        smoothed_greedy = smooth_data(greedy_data, sigma=1.5)
        smoothed_random = smooth_data(random_data, sigma=1.5)
        
        episodes = np.arange(1, min_length + 1)
        
        axins.plot(episodes, smoothed_greedy, color='orange', linewidth=2, marker='o', markersize=4, markevery=2, label='Greedy', alpha=0.8)
        axins.plot(episodes, smoothed_random, color='black', linewidth=2, marker='x', markersize=4, markevery=2, label='Random', alpha=0.8)
        
        # Set appropriate y-axis limits based on metric type
        if 'Energy Efficiency' in metric_name:
            axins.set_ylim(-2.5, 5)  # Fixed y-axis range for energy efficiency
        elif 'Sum Rate' in metric_name:
            # For sum rate, use data-dependent range but ensure it's visible
            greedy_range = np.ptp(smoothed_greedy)
            random_range = np.ptp(smoothed_random)
            max_range = max(greedy_range, random_range)
            if max_range < 0.1:  # If range is very small, use fixed range
                axins.set_ylim(0, 0.2)
            else:
                y_min = min(np.min(smoothed_greedy), np.min(smoothed_random)) * 0.9
                y_max = max(np.max(smoothed_greedy), np.max(smoothed_random)) * 1.1
                axins.set_ylim(y_min, y_max)
        elif 'Energy Harvesting' in metric_name:
            # For energy harvesting, use a smaller fixed range for better visibility
            axins.set_ylim(0, 0.0003)  # Fixed smaller y-axis range for energy harvesting
            # Use same mJ formatting for inset
            axins.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1000:.1f}'))
        
        axins.set_xlim(80, 90)  # Fixed range from episode 80 to 90
        
        # Set specific x-axis ticks
        axins.set_xticks([80, 85, 90])
        
        # Set tick label sizes for inset
        axins.tick_params(axis='both', which='major', labelsize=12)

        # Customize inset
        axins.grid(True, color='lightgray', linewidth=0.5)
        
        # Add connecting lines to show the zoomed area
        ax.indicate_inset_zoom(axins, edgecolor='black', alpha=0.5)
    
    # Adjust layout to accommodate legend outside the plot
    plt.subplots_adjust(right=0.85)
    return fig

# Prepare data for each environment
def prepare_energy_efficiency_data():
    # Simple environment data
    simple_data = {
        'MADDPG': ee_ddpg_simple,
        'MATD3': ee_td3_simple,
        'MASAC': ee_sac_simple,
        'MAPPO': ee_ppo_simple,
        'Greedy': ee_greedy_simple,
        'Random': ee_random_simple
    }
    
    # EGC environment data
    egc_data = {
        'MADDPG': ee_ddpg_egc,
        'MATD3': ee_td3_egc,
        'MASAC': ee_sac_egc,
        'MAPPO': ee_ppo_egc,
        'Greedy': ee_greedy_egc,
        'Random': ee_random_egc
    }
    
    # MRC environment data
    mrc_data = {
        'MADDPG': ee_ddpg_mrc,
        'MATD3': ee_td3_mrc,
        'MASAC': ee_sac_mrc,
        'MAPPO': ee_ppo_mrc,
        'Greedy': ee_greedy_mrc,
        'Random': ee_random_mrc
    }
    
    # SC environment data
    sc_data = {
        'MADDPG': ee_ddpg_sc,
        'MATD3': ee_td3_sc,
        'MASAC': ee_sac_sc,
        'MAPPO': ee_ppo_sc,
        'Greedy': ee_greedy_sc,
        'Random': ee_random_sc
    }
    
    return {
        'Simple': simple_data,
        'EGC': egc_data,
        'MRC': mrc_data,
        'SC': sc_data
    }

def prepare_sum_rate_data():
    # Only EGC environment data from plots_sr_eh.py
    egc_data = {
        'MADDPG': sr_ddpg_egc,
        'MATD3': sr_td3_egc,
        'MASAC': sr_sac_egc,
        'MAPPO': sr_ppo_egc,
        'Greedy': sr_greedy_egc,
        'Random': sr_random_egc
    }
    
    return {
        'EGC': egc_data
    }

def prepare_energy_harvesting_data():
    # Only EGC environment data from plots_sr_eh.py
    egc_data = {
        'MADDPG': eh_ddpg_egc,
        'MATD3': eh_td3_egc,
        'MASAC': eh_sac_egc,
        'MAPPO': eh_ppo_egc,
        'Greedy': eh_greedy_egc,
        'Random': eh_random_egc
    }
    
    return {
        'EGC': egc_data
    }

def main():
    # Get environment data for all metrics
    ee_data = prepare_energy_efficiency_data()
    sr_data = prepare_sum_rate_data()
    eh_data = prepare_energy_harvesting_data()
    
    # Create plots for each environment and metric
    for env_name in ['Simple', 'EGC', 'MRC', 'SC']:
        print(f"\nProcessing {env_name} environment...")
        
        # Energy Efficiency plots for all environments
        print(f"Creating energy efficiency plot for {env_name}...")
        fig_ee = plot_metric_comparison(env_name, ee_data[env_name], env_name, 
                                       "Energy Efficiency", "Average Energy Efficiency (b/J)")
        filename_ee_png = f'energy_efficiency_{env_name.lower()}_environment.png'
        filename_ee_eps = f'energy_efficiency_{env_name.lower()}_environment.eps'
        fig_ee.savefig(filename_ee_png, dpi=300, bbox_inches='tight')
        fig_ee.savefig(filename_ee_eps, format='eps', bbox_inches='tight')
        print(f"Saved: {filename_ee_png} and {filename_ee_eps}")
        plt.close(fig_ee)
    
    # Sum Rate plots only for EGC environment
    print(f"\nCreating sum rate plot for EGC environment...")
    fig_sr = plot_metric_comparison('EGC', sr_data['EGC'], 'EGC', 
                                   "Sum Rate", "Sum Rate")
    filename_sr_png = f'sum_rate_egc_environment.png'
    filename_sr_eps = f'sum_rate_egc_environment.eps'
    fig_sr.savefig(filename_sr_png, dpi=300, bbox_inches='tight')
    fig_sr.savefig(filename_sr_eps, format='eps', bbox_inches='tight')
    print(f"Saved: {filename_sr_png} and {filename_sr_eps}")
    plt.close(fig_sr)
    
    # Energy Harvesting plots only for EGC environment
    print(f"\nCreating energy harvesting plot for EGC environment...")
    fig_eh = plot_metric_comparison('EGC', eh_data['EGC'], 'EGC', 
                                   "Energy Harvesting", "Energy Harvesting")
    filename_eh_png = f'energy_harvesting_egc_environment.png'
    filename_eh_eps = f'energy_harvesting_egc_environment.eps'
    fig_eh.savefig(filename_eh_png, dpi=300, bbox_inches='tight')
    fig_eh.savefig(filename_eh_eps, format='eps', bbox_inches='tight')
    print(f"Saved: {filename_eh_png} and {filename_eh_eps}")
    plt.close(fig_eh)
    
    print("\n" + "="*50)
    print("All plots have been created successfully!")
    print("="*50)
    print("Energy Efficiency plots (PNG & EPS) for all environments:")
    print("- energy_efficiency_simple_environment.png/.eps")
    print("- energy_efficiency_egc_environment.png/.eps") 
    print("- energy_efficiency_mrc_environment.png/.eps")
    print("- energy_efficiency_sc_environment.png/.eps")
    print("\nSum Rate plot (PNG & EPS) for EGC environment only:")
    print("- sum_rate_egc_environment.png/.eps")
    print("\nEnergy Harvesting plot (PNG & EPS) for EGC environment only:")
    print("- energy_harvesting_egc_environment.png/.eps")
    print("="*50)

if __name__ == "__main__":
    main() 