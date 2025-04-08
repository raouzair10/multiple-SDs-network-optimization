import numpy as np
import torch
import time
import random
from env_simple import Env_cellular as env_simple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from ddpg import Agent as DDPG_AGENT
import warnings;
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch, Rectangle
##################### System Parameters ####################
Emax = 0.3              # Maximum battery capacity for SD
num_PDs = 2             # Number of PDs
num_SDs = 1             # Number of SDs
L = 2                   # Number of antennas on the SD
T = 1                   # Time duration for each time slot
eta = 0.7               # Energy harvesting efficiency (between 0 and 1)
Pn = 1                  # transmit power of PDs
Pmax = 0.2              # Maximum transmit power of the SD
w_csk = 0.000003        # circuit and signal processing power for the SD
# w_d = 1.5 * (10 ** -5)  # Processing power for diversity combining
# w_egc = 1 * (10 ** -6)  # Processing power for EGC
# w_mrc = 2 * (10 ** -6)  # Processing power for MRC

##################### Hyper Parameters #####################
MAX_EPISODES = 3
MAX_EP_STEPS = 5
LR_A = 0.0002    # learning rate for actor
LR_C = 0.0004    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
s_dim = 3       # dimsion of states
a_dim = 1       # dimension of action
a_bound = 1     # bound of action
state_am = 1000
verbose = True

##################### Set random seed number #####################
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

##################### User Input for Number of SDs ####################
# num_SDs = int(input("Enter the number of Secondary Devices (at least 1): "))

###################### Network Nodes Deployment  #######################

# Randomly generate PD and SD locations within a specific range (0 to 10 for both x and y)
location_PDs = np.random.uniform(low=0, high=1000, size=(num_PDs, 2)).astype(int)
location_SDs = np.random.uniform(low=0, high=10, size=(num_SDs, 2)).astype(int)
# location_PDs = np.array(([0,1],[0,1000]))
# location_PDs = [[5 7]
#                 [6 5]]
# location_SDs = [[4 6]
#                 [4 8]]

############################### Fadings ################################

#### fixed fading value for the SD, set to 1 for simplicity ####
fading_0 = 1

# # hnx[0, 0] = 0.78954 is the fading value for the first PD on the first path
# hnx = np.array([[0.78954, 0.62134, 0.62134, 0.75324, 0.98921, 0.85324, 0.95324, 0.65324, 0.79324, 0.92324],
#                 [0.75324, 0.77954, 0.85324, 0.75324, 0.95324, 0.88324, 0.81324, 0.80324, 0.65324, 0.95324]])

# Generate hnx randomly for each PD and each path
hnx = np.random.uniform(low=0.6, high=0.99, size=(num_PDs, L))
hnx = np.around(hnx, decimals=5)

# hnx[0, 0] = 0.77066 is the fading value for the first PD on the first path
# hnx = [[0.77066 0.94779]
#        [0.97583 0.74954]]

# Gains between PD and Base Station
hnPD = np.array([0.79643, 0.87451])

# placeholder array initialized with zeros to store aggregated fading values for each PD
fading_n_ = np.zeros(num_PDs)

##################### Gains Passing Approach ##########################
for i in range(num_PDs):
    fading_n_[i] = np.random.choice(hnx[i])     # TODO: REMOVE NP AFTER TESTING
# fading_n_ = [0.77066 0.74954]

fading_n = np.vstack((fading_n_, hnPD))
fading_n = np.matrix.transpose(fading_n)
# fading_n = [[0.77066 0.79643]
#             [0.74954 0.87451]]
# fading_n = [[PD1-SD1 PD1-BS]
#             [PD2-SD1 PD2-BS]]

myenv = env_simple(MAX_EP_STEPS, s_dim, location_PDs, location_SDs, Emax, num_PDs, T, eta, Pn, Pmax, w_csk, fading_n, fading_0)

#---------------------------------Initializing DDPG Agent--------------------------------------------------------------
ddpg_agent=DDPG_AGENT(LR_A,LR_C,s_dim,TAU,a_bound,GAMMA,n_actions=1,max_size=MEMORY_CAPACITY,batch_size=BATCH_SIZE)
#-----------------------------------------------------------------------------------------------------------------------

var = 1  # control exploration

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
    batter_ini = myenv.reset()                   # batter_ini is 0.3
    s = myenv.hn[i % myenv.num_PDs, :].tolist()  # the current PD's gains, 2 element [PDn-SD1, PDn-BS] # TODO: replaced channel_sequence with hn
    s.append(batter_ini)
    s = np.reshape(s, (1, s_dim))
    s = s * state_am    # amplify the state
    # s    is   [[1.18504087e-03 5.38452764e-01 3.00000000e+02]] (for even i)
    # s is like [[PDn-SD1, PDn-BS batter_ini]] (amplified)

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

    s_traj_ddpg = []
    s_traj_random = []
    s_traj_greedy = []

    for j in range(MAX_EP_STEPS):
        ######################## DDPG ########################
        a_ddpg = ddpg_agent.choose_action(s_ddpg)
        a_ddpg = np.clip(np.random.normal(a_ddpg, var), 0, 1)  # add randomness to action selection for exploration
        r_ddpg, s_ddpg_, EHD, done, ee_ddpg = myenv.step(a_ddpg, s_ddpg / state_am, j+i) # (action, state, timestep)
        s_ddpg_ = s_ddpg_ * state_am
        s_traj_ddpg.append(s_ddpg_)
        ddpg_agent.remember(s_ddpg[0], a_ddpg, r_ddpg, s_ddpg_[0], 0)
        ep_reward_ddpg += int(*r_ddpg)
        eh_reward_ddpg += EHD
        ep_ee_ddpg += ee_ddpg
        ######################## Greedy ########################
        r_greedy, s_next_greedy, EHG, done, ee_greedy = myenv.step_greedy(s_greedy / state_am, j)
        s_traj_greedy.append(s_next_greedy)
        s_greedy = s_next_greedy * state_am
        ep_reward_greedy += (r_greedy)
        eh_reward_greedy += EHG
        ep_ee_greedy += ee_greedy
        ######################## Random ########################
        r_random, s_next_random, EHR, done, ee_random = myenv.step_random(s_random / state_am, j)
        s_traj_random.append(s_next_random)
        s_random = s_next_random * state_am
        ep_reward_random += r_random
        eh_reward_random += EHR
        ep_ee_random += ee_random
        ########################### Network Trainings ######################
        if var > 0.1:
            var *= .9998  # decay the action randomness
        # ----------------DDPG ALGORITHM TRAINING-----------------------
        ddpg_agent.learn()
        ########################### Update States ######################
        s_ddpg = s_ddpg_

        if j == MAX_EP_STEPS-1:
            print('Avg EE for Episode:', i, ' DDPG: %i' % int(ep_ee_ddpg/MAX_EP_STEPS), 'Greedy: %i' % int(ep_ee_greedy/MAX_EP_STEPS),  'Random: %i' % int(ep_ee_random/MAX_EP_STEPS))
    
    ep_reward_ddpg = np.reshape(ep_reward_ddpg/MAX_EP_STEPS, (1,))
    ep_rewardall_ddpg.append(ep_reward_ddpg)
    eh_reward_ddpg = np.reshape(eh_reward_ddpg / MAX_EP_STEPS, (1,))
    eh_rewardall_ddpg.append(eh_reward_ddpg)
    ep_ee_ddpg = np.reshape(ep_ee_ddpg/MAX_EP_STEPS, (1,))
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

print("=================================================================================")
# ########### PLOTTING FIGURES ###############
# fig, ax = plt.subplots()
# ax.plot(ep_rewardall_ddpg, "^-", label='DDPG', linewidth=0.75 , color= 'darkblue')
# ax.plot(ep_rewardall_greedy, "o-", label='Greedy', linewidth=0.75)
# ax.plot(ep_rewardall_random, "x-", label='Random', color='black', linewidth=0.75)
# ax.set_xlabel("Episodes")
# ax.set_ylabel("Epsiodic Rewards")
# ax.legend()
# ax.margins(x=0)
# ax.set_xlim(1, MAX_EPISODES-1)
# ax.grid(which="both", axis='y', linestyle=':', color='lightgray', linewidth=0.5)
# ax.minorticks_on()
# ax.tick_params(which="minor", bottom=False, left=False)

# # Make the legend draggable
# legend1 = ax.legend()
# legend1.set_draggable(True)

# fig.savefig('Fig1.eps', format='eps', bbox_inches='tight')
# fig.savefig('Fig1.png', format='png', bbox_inches='tight')
# plt.show()

# # Plot for energy harvested
# fig2, ax = plt.subplots()
# ax.plot(eh_rewardall_ddpg, "^-", label='DDPG', linewidth=0.75 , color= 'darkblue')
# ax.plot(eh_rewardall_greedy, "o-", label='Greedy', linewidth=0.75)
# ax.plot(eh_rewardall_random, "x-", label='Random', color='black', linewidth=0.75)
# ax.set_xlabel("Episodes")
# ax.set_ylabel("Energy Harvested (J)")
# ax.legend()
# ax.margins(x=0)
# ax.set_xlim(1, MAX_EPISODES-1)
# ax.set_yscale('log')
# ax.grid(which="both", axis='y', linestyle=':', color='lightgray', linewidth=0.5)
# ax.minorticks_on()
# ax.tick_params(which="minor", bottom=False, left=False)

# # Make the legend draggable
# legend2 = ax.legend()
# legend2.set_draggable(True)

# fig2.savefig('Fig2.eps', format='eps', bbox_inches='tight')
# fig2.savefig('Fig2.png', format='png', bbox_inches='tight')

# plt.show()

# fig3, ax = plt.subplots()
# ax.plot(ee_rewardall_ddpg, "^-", label='DDPG', linewidth=0.75 , color= 'darkblue')
# ax.plot(ee_rewardall_greedy, "o-", label='Greedy', linewidth=0.75, color= 'orange')
# ax.plot(ee_rewardall_random, "x-", label='Random', color='black', linewidth=0.75)
# ax.set_xlabel("Episodes")
# ax.set_ylabel("Average Energy Efficiency (b/J)")
# ax.legend()
# ax.margins(x=0)
# ax.set_xlim(0, MAX_EPISODES)
# ax.grid(which="both", axis='y', linestyle=':', color='lightgray', linewidth=0.5)
# ax.minorticks_on()
# ax.tick_params(which="minor", bottom=False, left=False)

# # Add a zoomed-in inset plot for Greedy and Random comparison
# axins = inset_axes(ax, width="18%", height="13%", loc=4, borderpad=4)  # Adjust size and location
# axins.plot(ee_rewardall_greedy, "o-", label='Greedy', color='orange', linewidth=0.75)
# axins.plot(ee_rewardall_random, "x-", label='Random', color='black', linewidth=0.75)

# # Set limits for the zoomed-in section that matches the data in the main plot
# x1, x2 = 165, 175 # Adjust based on your data range for episodes
# y1, y2 = 40, 60  # Adjust to fit the energy efficiency range
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)

# # Add grid and adjust ticks for the inset
# axins.grid(which="both", axis='y', linestyle=':', color='lightgray', linewidth=0.5)
# axins.set_xticks([165, 170, 175])  # Adjust to your actual data range
# axins.set_yticks([45, 50, 55])    # Adjust y-axis ticks based on zoomed data

# # Highlight the zoomed region on the main plot with a rectangle
# rect = Rectangle((x1, -5000), x2-x1, 10000, linewidth=1, edgecolor='black', facecolor='none', linestyle='--')
# ax.add_patch(rect)

# con = ConnectionPatch(xyA=((x1 + x2) / 2, (y1 + y2) / 2 + 5000 ), coordsA=ax.transData,
#                       xyB=(x1 + 4, y1), coordsB=axins.transData,
#                       arrowstyle='->', color='black', lw=1)

# axins.add_artist(con)


# # Move the legend to the bottom left
# legend3 = ax.legend(loc='lower left', bbox_to_anchor=(0.1, 0.1), edgecolor="black")
# legend3.get_frame().set_alpha(None)
# legend3.get_frame().set_facecolor((1, 1, 1, 0))
# legend3.set_draggable(True)

# # Save figures with different formats
# fig3.savefig('Fig3.eps', format='eps', bbox_inches='tight')
# fig3.savefig('Fig3.png', format='png', bbox_inches='tight')

# plt.show()