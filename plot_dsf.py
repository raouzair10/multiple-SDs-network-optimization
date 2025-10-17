import matplotlib.pyplot as plt
import numpy as np

# Set global font sizes (optional, but good practice)
plt.rcParams['font.size'] = 16 
plt.rcParams['font.family'] = 'Times New Roman'

# Import DSF data arrays from plots_dsf.py
from plots_dsf import (
    sd0_ddpg, sd1_ddpg,
    sd0_td3, sd1_td3,
    sd0_ppo, sd1_ppo,
    sd0_sac, sd1_sac,
    sd0_random, sd1_random,
    sd0_greedy, sd1_greedy
)

# Create episodes array (1-200)
episodes = list(range(1, 201))

# Define algorithms and their data
algorithms = {
    'maddpg': (sd0_ddpg, sd1_ddpg),
    'matd3': (sd0_td3, sd1_td3),
    'mappo': (sd0_ppo, sd1_ppo),
    'masac': (sd0_sac, sd1_sac),
    'random': (sd0_random, sd1_random),
    'greedy': (sd0_greedy, sd1_greedy)
}

# Loop through each algorithm and create individual plots
for alg_name, (sd0_data, sd1_data) in algorithms.items():
    # Create a new figure
    plt.figure(figsize=(10, 6))
    
    # Plot SU 1 (red line) and SU 2 (blue line) with markers at regular intervals
    plt.plot(episodes, sd0_data, 'r-', label='SU 1', linewidth=2, marker='o', markevery=20, markersize=8)
    plt.plot(episodes, sd1_data, 'b-', label='SU 2', linewidth=2, marker='s', markevery=20, markersize=8)
    
    # Set axis limits to 200x200
    plt.xlim(1, 200)
    plt.ylim(0, 200)
    
    # Set labels - UPDATED FONTSIZE TO 26
    plt.xlabel('Episodes', fontsize=30)
    plt.ylabel('Device Selection Frequency', fontsize=30)
    
    # Add grid with lightgray color and thin lines
    plt.grid(True, linestyle=':', color='lightgray', linewidth=0.2)
    
    # Add legend - UPDATED FONTSIZE TO 18
    plt.legend(fontsize=22)
    
    # You might also want to increase tick label size:
    plt.tick_params(axis='both', which='major', labelsize=24) 
    
    # Save as PNG
    plt.savefig(f'dsf_{alg_name}.png', dpi=300, bbox_inches='tight')
    
    # Save as EPS
    plt.savefig(f'dsf_{alg_name}.eps', format='eps', dpi=300, bbox_inches='tight')
    
    # Close the figure (don't display)
    plt.close()

# Create fairness comparison bar chart
plt.figure(figsize=(12, 8))

# Calculate average selections for each SU across all episodes
fairness_data = {
    'MADDPG': [np.mean(sd0_ddpg), np.mean(sd1_ddpg)],
    'MATD3': [np.mean(sd0_td3), np.mean(sd1_td3)],
    'MAPPO': [np.mean(sd0_ppo), np.mean(sd1_ppo)],
    'MASAC': [np.mean(sd0_sac), np.mean(sd1_sac)],
    'Greedy': [np.mean(sd0_greedy), np.mean(sd1_greedy)],
    'Random': [np.mean(sd0_random), np.mean(sd1_random)]
}

# Set up bar positions
algorithms = list(fairness_data.keys())
x = np.arange(len(algorithms))
width = 0.35

# Create bars
su1_bars = plt.bar(x - width/2, [fairness_data[alg][0] for alg in algorithms], 
                     width, label='SU 1', color='red', alpha=0.8)

# SU2 bars with slanted line hatch - ADDED HATCH PARAMETER
su2_bars = plt.bar(x + width/2, [fairness_data[alg][1] for alg in algorithms], 
                     width, label='SU 2', color='blue', alpha=0.8, hatch='//', edgecolor='black') # Added hatch and edgecolor for visibility

# Customize the plot
plt.xlabel('Algorithms', fontsize=30)
plt.ylabel('Number of times SU is selected', fontsize=30)
plt.xticks(x, algorithms, fontsize=24) 
plt.ylim(0, 200)  # Set y-axis limit to 200

plt.legend(fontsize=22)

plt.yticks(fontsize=24)

# Save fairness plot
plt.savefig('fairness_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('fairness_comparison.eps', format='eps', dpi=300, bbox_inches='tight')
plt.close()

print("All DSF plots and fairness comparison plot have been created and saved as PNG and EPS files.")