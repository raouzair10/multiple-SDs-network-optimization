import numpy as np
import torch
import time
import random
from env_multiple_sd import Env_cellular as env_multiple_sd #replace with your new env file
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from maddpg import Agent as DDPG_AGENT
import warnings;
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch, Rectangle

##################### System Parameters ####################
Emax = 0.5
num_PDs = 2
num_SDs = 2 # Set to the desired number of SDs
L = 2
T = 1
eta = 0.7
Pn = 1
Pmax = 0.2
w_csk = 0.000003

##################### Hyper Parameters #####################
MAX_EPISODES = 200
MAX_EP_STEPS = 350
LR_A = 0.0002
LR_C = 0.0004
GAMMA = 0.9
TAU = 0.01
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
s_dim = (3 * num_SDs) + 1 # Calculate state dimension
a_dim = 2 # Calculate action dimension
a_bound = 1
state_am = 1000

##################### Set random seed number #####################
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

##################### Network Nodes Deployment #####################
location_PDs = np.random.uniform(low=0, high=1000, size=(num_PDs, 2)).astype(int)
# [[548 715]
#  [602 544]]
location_SDs = np.random.uniform(low=0, high=10, size=(num_SDs, 2)).astype(int)
# [[4 6]
#  [4 8]]

############################### Fadings ################################
fading_SD_BS = np.ones(num_SDs)     # fading between SDs and BS
fading_PD_BS = np.random.uniform(low=0.6, high=0.99, size=(num_PDs))  # Fading between PDs and Base Station
fading_PD_BS = np.around(fading_PD_BS, decimals=5)
hnx = np.random.uniform(low=0.6, high=0.99, size=(num_PDs, num_SDs, L)) # fading values for PDs and antennas on SDs
hnx = np.around(hnx, decimals=5)
# [[[0.97583 0.74954]
#   [0.90877 0.80627]]
#  [[0.82154 0.96098]
#   [0.6277  0.63398]]]

# [[[PD1-SD1-L1 PD1-SD1-L2]
#   [PD1-SD2-L1 PD1-SD2-L2]]
#  [[PD2-SD1-L1 PD2-SD1-L2]
#   [PD2-SD2-L1  PD2-SD2-L2]]]

fading_PD_SD = np.zeros((num_PDs, num_SDs))

for i in range(num_PDs):
    for j in range(num_SDs):
        fading_PD_SD[i][j] = np.random.choice(hnx[i][j])
# [[0.74954 0.90877]
#  [0.96098 0.63398]]
# [[PD1-SD1 PD1-SD2]
#  [PD2-SD1 PD2SD2]]

myenv = env_multiple_sd(MAX_EP_STEPS, s_dim, location_PDs, location_SDs, Emax, num_PDs, T, eta, Pn, Pmax, w_csk, fading_PD_SD, fading_PD_BS, fading_SD_BS, num_SDs)

#---------------------------------Initializing DDPG Agent--------------------------------------------------------------
ddpg_agent = DDPG_AGENT(LR_A, LR_C, s_dim, TAU, GAMMA, n_actions=a_dim, max_size=MEMORY_CAPACITY, batch_size=BATCH_SIZE, num_agents=num_SDs)
#-----------------------------------------------------------------------------------------------------------------------

var = 1

ep_rewardall_ddpg = []
ep_rewardall_greedy = []
ep_rewardall_random = []

eh_rewardall_ddpg = []
eh_rewardall_greedy = []
eh_rewardall_random = []

ee_rewardall_ddpg = []
ee_rewardall_greedy = []
ee_rewardall_random = []

for i in range(MAX_EPISODES):
    batter_ini = myenv.reset()
    s = []
    for sd in range(num_SDs):
        s.append(myenv.hn_PD_SD[i % myenv.num_PDs, sd])
        s.append(myenv.hn_SD_BS[sd])
        s.append(batter_ini[sd])
    s.append(myenv.hn_PD_BS[i % myenv.num_PDs])
    s = np.reshape(s, (1, s_dim))   
    # [[7.10046823e-13 1.80299692e-06 3.00000000e-01 1.24827307e-12 9.44854682e-07 3.00000000e-01 7.36531872e-13]]
    # [[hn-SD1-PDi hn-SD1-BS battery-SD1 hn-SD2-PDi hn-SD2-BS battery-SD2 hn-PDi-BS]]
    s = s * state_am

    s_ddpg = s
    s_greedy = s
    s_random = s

    ep_ee_ddpg = 0
    ep_ee_random = 0
    ep_ee_greedy = 0

    ep_reward_ddpg = 0
    ep_reward_random = 0
    ep_reward_greedy = 0

    eh_reward_ddpg = 0
    eh_reward_random = 0
    eh_reward_greedy = 0

    for j in range(MAX_EP_STEPS):
        ######################## DDPG ########################
        a_ddpg = list(ddpg_agent.choose_action(s_ddpg))
        a_ddpg[1] = np.clip(np.random.normal(a_ddpg[1], var), 0, 1)[0]

        r_ddpg, s_ddpg_, EHD, ee_ddpg = myenv.step(a_ddpg, s_ddpg / state_am, j + i)
        s_ddpg_ = s_ddpg_ * state_am
        ddpg_agent.remember(s_ddpg[0], a_ddpg, r_ddpg, s_ddpg_[0])
        ep_reward_ddpg += r_ddpg
        eh_reward_ddpg += EHD
        ep_ee_ddpg += ee_ddpg

        # ######################## Greedy ########################
        # r_greedy, s_next_greedy, EHG, done, ee_greedy = myenv.step_greedy(s_greedy / state_am, j)
        # s_greedy = s_next_greedy * state_am
        # ep_reward_greedy += sum(r_greedy)
        # eh_reward_greedy += sum(EHG)
        # ep_ee_greedy += sum(ee_greedy)

        # ######################## Random ########################
        # r_random, s_next_random, EHR, done, ee_random = myenv.step_random(s_random / state_am, j)
        # s_random = s_next_random * state_am
        # ep_reward_random += sum(r_random)
        # eh_reward_random += sum(EHR)
        # ep_ee_random += sum(ee_random)

        if var > 0.1:
            var *= .9998
        ddpg_agent.learn()
        s_ddpg = s_ddpg_

        if j == MAX_EP_STEPS - 1:
            print('Avg EE for Episode:', i, ' DDPG: %i' % int(ep_ee_ddpg / MAX_EP_STEPS), 'Greedy: %i' % int(ep_ee_greedy / MAX_EP_STEPS), 'Random: %i' % int(ep_ee_random / MAX_EP_STEPS))
    ep_reward_ddpg = np.reshape(ep_reward_ddpg / MAX_EP_STEPS, (1,))
    ep_rewardall_ddpg.append(ep_reward_ddpg)
    eh_reward_ddpg = np.reshape(eh_reward_ddpg / MAX_EP_STEPS, (1,))
    eh_rewardall_ddpg.append(eh_reward_ddpg)
    ep_ee_ddpg = np.reshape(ep_ee_ddpg / MAX_EP_STEPS, (1,))
    ee_rewardall_ddpg.append(ep_ee_ddpg)
    
    ep_reward_greedy = np.reshape(ep_reward_greedy/MAX_EP_STEPS, (1,))
    ep_rewardall_greedy.append(ep_reward_greedy)
    eh_reward_greedy = np.reshape(eh_reward_greedy / MAX_EP_STEPS, (1,))
    eh_rewardall_greedy.append(eh_reward_greedy)
    ep_ee_greedy = np.reshape(ep_ee_greedy/MAX_EP_STEPS, (1,))
    ee_rewardall_greedy.append(ep_ee_greedy)

    ep_reward_random = np.reshape(ep_reward_random/MAX_EP_STEPS, (1,))
    ep_rewardall_random.append(ep_reward_random)
    eh_reward_random = np.reshape(eh_reward_random / MAX_EP_STEPS, (1,))
    eh_rewardall_random.append(eh_reward_random)
    ep_ee_random = np.reshape(ep_ee_random/MAX_EP_STEPS, (1,))
    ee_rewardall_random.append(ep_ee_random)


# ######### CALCULATING AVERAGE REWARDS AND EH ##########
avg_rewards = [sum(ep_rewardall_ddpg)/len(ep_rewardall_ddpg),
                sum(ep_rewardall_greedy)/len(ep_rewardall_greedy),
                sum(ep_rewardall_random)/len(ep_rewardall_random)]

avg_harvested_energy = [sum(eh_rewardall_ddpg)/len(eh_rewardall_ddpg),
                        sum(eh_rewardall_greedy)/len(eh_rewardall_greedy),
                        sum(eh_rewardall_random)/len(eh_rewardall_random)]

# ########### PLOTTING FIGURES ###############
fig, ax = plt.subplots()
ax.plot(ep_rewardall_ddpg, "^-", label='DDPG', linewidth=0.75 , color= 'darkblue')
ax.plot(ep_rewardall_greedy, "o-", label='Greedy', linewidth=0.75)
ax.plot(ep_rewardall_random, "x-", label='Random', color='black', linewidth=0.75)
ax.set_xlabel("Episodes")
ax.set_ylabel("Epsiodic Rewards")
ax.legend()
ax.margins(x=0)
ax.set_xlim(1, MAX_EPISODES-1)
ax.grid(which="both", axis='y', linestyle=':', color='lightgray', linewidth=0.5)
ax.minorticks_on()
ax.tick_params(which="minor", bottom=False, left=False)

# Make the legend draggable
legend1 = ax.legend()
legend1.set_draggable(True)

fig.savefig('Fig1.eps', format='eps', bbox_inches='tight')
fig.savefig('Fig1.png', format='png', bbox_inches='tight')
plt.show()

# Plot for energy harvested
fig2, ax = plt.subplots()
ax.plot(eh_rewardall_ddpg, "^-", label='DDPG', linewidth=0.75 , color= 'darkblue')
ax.plot(eh_rewardall_greedy, "o-", label='Greedy', linewidth=0.75)
ax.plot(eh_rewardall_random, "x-", label='Random', color='black', linewidth=0.75)
ax.set_xlabel("Episodes")
ax.set_ylabel("Energy Harvested (J)")
ax.legend()
ax.margins(x=0)
ax.set_xlim(1, MAX_EPISODES-1)
ax.set_yscale('log')
ax.grid(which="both", axis='y', linestyle=':', color='lightgray', linewidth=0.5)
ax.minorticks_on()
ax.tick_params(which="minor", bottom=False, left=False)

# Make the legend draggable
legend2 = ax.legend()
legend2.set_draggable(True)

fig2.savefig('Fig2.eps', format='eps', bbox_inches='tight')
fig2.savefig('Fig2.png', format='png', bbox_inches='tight')

plt.show()

fig3, ax = plt.subplots()
ax.plot(ee_rewardall_ddpg, "^-", label='DDPG', linewidth=0.75 , color= 'darkblue')
ax.plot(ee_rewardall_greedy, "o-", label='Greedy', linewidth=0.75, color= 'orange')
ax.plot(ee_rewardall_random, "x-", label='Random', color='black', linewidth=0.75)
ax.set_xlabel("Episodes")
ax.set_ylabel("Average Energy Efficiency (b/J)")
ax.legend()
ax.margins(x=0)
ax.set_xlim(0, MAX_EPISODES)
ax.grid(which="both", axis='y', linestyle=':', color='lightgray', linewidth=0.5)
ax.minorticks_on()
ax.tick_params(which="minor", bottom=False, left=False)

# Add a zoomed-in inset plot for Greedy and Random comparison
axins = inset_axes(ax, width="18%", height="13%", loc=4, borderpad=4)  # Adjust size and location
axins.plot(ee_rewardall_greedy, "o-", label='Greedy', color='orange', linewidth=0.75)
axins.plot(ee_rewardall_random, "x-", label='Random', color='black', linewidth=0.75)

# Set limits for the zoomed-in section that matches the data in the main plot
x1, x2 = 165, 175 # Adjust based on your data range for episodes
y1, y2 = 40, 60  # Adjust to fit the energy efficiency range
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

# Add grid and adjust ticks for the inset
axins.grid(which="both", axis='y', linestyle=':', color='lightgray', linewidth=0.5)
axins.set_xticks([165, 170, 175])  # Adjust to your actual data range
axins.set_yticks([45, 50, 55])    # Adjust y-axis ticks based on zoomed data

# Highlight the zoomed region on the main plot with a rectangle
rect = Rectangle((x1, -5000), x2-x1, 10000, linewidth=1, edgecolor='black', facecolor='none', linestyle='--')
ax.add_patch(rect)

con = ConnectionPatch(xyA=((x1 + x2) / 2, (y1 + y2) / 2 + 5000 ), coordsA=ax.transData,
                      xyB=(x1 + 4, y1), coordsB=axins.transData,
                      arrowstyle='->', color='black', lw=1)

axins.add_artist(con)


# Move the legend to the bottom left
legend3 = ax.legend(loc='lower left', bbox_to_anchor=(0.1, 0.1), edgecolor="black")
legend3.get_frame().set_alpha(None)
legend3.get_frame().set_facecolor((1, 1, 1, 0))
legend3.set_draggable(True)

# Save figures with different formats
fig3.savefig('Fig3.eps', format='eps', bbox_inches='tight')
fig3.savefig('Fig3.png', format='png', bbox_inches='tight')

plt.show()