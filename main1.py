import numpy as np
import torch
import random
from env_simple import Env_cellular as env_simple
from env_egc import Env_cellular as env_egc
from env_sc import Env_cellular as env_sc
from env_mrc import Env_cellular as env_mrc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from maddpg import Agent as MADDPG
from matd3 import MATD3
from mappo import MAPPO
from masac import MASACAgent
import warnings;
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch, Rectangle

##################### System Parameters ####################
Emax = 0.3
num_PDs = 2
num_SDs = 2 # Set to the desired number of SDs
L = 2
T = 1
eta = 0.7
Pn = 1
Pmax = 0.2
w_csk = 0.000003
w_d = 1.5 * (10 ** -5)
w_egc = 1 * (10 ** -6)
w_mrc = 2 * (10 ** -6)

##################### Hyper Parameters #####################
MAX_EPISODES = 200
MAX_EP_STEPS = 300
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

##################### Hyperparameters for PPO #####################
has_continuous_action_space = True  # continuous action space; else discrete
action_std = 0.6                      # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05          # linearly decay action_std
min_action_std = 0.1                  # minimum action_std
action_std_decay_freq = int(2.5e6)    # action_std decay frequency (in num timesteps)
update_timestep_ppo = MAX_EP_STEPS * 3    # update policy every n timesteps
K_epochs_ppo = 40                         # update policy for K epochs in one PPO update
eps_clip_ppo = 0.2                        # clip parameter for PPO
a_dim_ppo = 1 # Action dimension per agent (time-sharing fraction)
s_dim_ppo = 4 # Local state dimension for each SD [hn_PD_SD, hn_SD_BS, battery_level, hn_PD_BS]

##################### Hyperparameters for MATD3 #####################
LR_A_MATD3 = 0.0002
LR_C_MATD3 = 0.0004
GAMMA_MATD3 = 0.9
TAU_MATD3 = 0.01
MEMORY_CAPACITY_MATD3 = 10000
BATCH_SIZE_MATD3 = 32
UPDATE_ACTOR_INTERVAL_MATD3 = 2

##################### Hyperparameters for MASAC #####################
LR_A_MASAC = 0.0002
LR_C_MASAC = 0.0004
GAMMA_MASAC = 0.9
TAU_MASAC = 0.01
BUFFER_CAPACITY_MASAC = 10000
BATCH_SIZE_MASAC = 32
ALPHA_MASAC = 0.2  # Initial alpha
AUTO_ALPHA_MASAC = True
TARGET_ENTROPY_MASAC = -torch.prod(torch.tensor(1, dtype=torch.float32)).item() # Adjust if action space is different


##################### Set random seed number #####################
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

##################### Network Nodes Deployment #####################
location_PDs = np.array([[0, 1],[0,1000]])
location_SDs = np.array([[1,1],[2,2]])

# location_PDs = np.random.uniform(low=0, high=1000, size=(num_PDs, 2)).astype(int)
# # [[548 715]
# #  [602 544]]
# location_SDs = np.random.uniform(low=0, high=10, size=(num_SDs, 2)).astype(int)
# # [[4 6]
# #  [4 8]]

############################### Fadings ################################
fading_SD_BS = np.ones(num_SDs)     # fading between SDs and BS

fading_PD_BS = np.array([0.79643, 0.87451])
# fading_PD_BS = np.random.uniform(low=0.6, high=0.99, size=(num_PDs))  # Fading between PDs and Base Station
# fading_PD_BS = np.around(fading_PD_BS, decimals=5)

# hnx = np.array([[[0.78954, 0.62134]], [[0.75324, 0.77954]]])
hnx = np.array([[[0.90877, 0.80627],
  [0.82154, 0.96098]],
 [[0.6277, 0.63398],
  [0.60789, 0.92472]]])
# hnx = np.random.uniform(low=0.6, high=0.99, size=(num_PDs, num_SDs, L)) # fading values for PDs and antennas on SDs
# hnx = np.around(hnx, decimals=5)
# [[[0.90877 0.80627]
#   [0.82154 0.96098]]
#  [[0.6277  0.63398]
#   [0.60789 0.92472]]]
# [[[PD1-SD1-L1 PD1-SD1-L2]
#   [PD1-SD2-L1 PD1-SD2-L2]]
#  [[PD2-SD1-L1 PD2-SD1-L2]
#   [PD2-SD2-L1  PD2-SD2-L2]]]

fading_PD_SD = np.zeros((num_PDs, num_SDs))

diversity_mode = int(input("Enter the diversity technique to harvest energy (0:Simple, 1:EGC, 2:MRC, 3:SC): "))

if diversity_mode == 1: # EGC
    for i in range(num_PDs):
        for j in range(num_SDs):
            fading_PD_SD[i][j] = np.sum(hnx[i][j]) ** 2 # square of sum
    myenv = env_egc(MAX_EP_STEPS, s_dim, location_PDs, location_SDs, Emax, num_PDs, L, T, eta, Pn, Pmax, w_d, w_egc, w_csk, fading_PD_SD, fading_PD_BS, fading_SD_BS, num_SDs)

elif diversity_mode == 2:   # MRC
    for i in range(num_PDs):
        for j in range(num_SDs):
            fading_PD_SD[i][j] = np.sum(hnx[i][j] ** 2) # sum of squares
    myenv = env_mrc(MAX_EP_STEPS, s_dim, location_PDs, location_SDs, Emax, num_PDs, L, T, eta, Pn, Pmax, w_d, w_egc, w_csk, fading_PD_SD, fading_PD_BS, fading_SD_BS, num_SDs)

elif diversity_mode == 3:   # SC
    for i in range(num_PDs):
        for j in range(num_SDs):
            fading_PD_SD[i][j] = np.max(hnx[i][j] ** 2) # max squared value
    myenv = env_sc(MAX_EP_STEPS, s_dim, location_PDs, location_SDs, Emax, num_PDs, T, eta, Pn, Pmax, w_d, w_csk, fading_PD_SD, fading_PD_BS, fading_SD_BS, num_SDs)


else:   # simple (no diversity) 
    for i in range(num_PDs):
        for j in range(num_SDs):
            fading_PD_SD[i][j] = random.choice(hnx[i][j]) # random value
            # [[0.90877 0.96098]
            #  [0.63398 0.60789]]
            # [[PD1-SD1 PD1-SD2]
            #  [PD2-SD1 PD2SD2]]
    myenv = env_simple(MAX_EP_STEPS, s_dim, location_PDs, location_SDs, Emax, num_PDs, T, eta, Pn, Pmax, w_csk, fading_PD_SD, fading_PD_BS, fading_SD_BS, num_SDs)

def choose_SD(s):
    sd_qualities = []
    for sd_index in range(num_SDs):
        sd = []
        sd.append(s[0][sd_index * 3])  # hn-SDj-PDi
        sd.append(s[0][sd_index * 3 + 1])  # hn-SDj-BS
        sd.append(s[0][sd_index * 3 + 2]) # battery level

        sd = np.array(sd)
        min_val = np.min(sd)
        max_val = np.max(sd)

        sd = (sd - min_val) / (max_val - min_val)
        sd_quality = np.sum(sd)
        sd_qualities.append(sd_quality)

    # Choose SD with best quality
    if np.random.rand() < 0.1: # 10% random selection.
        best_sd_index = np.random.randint(num_SDs)
    else:
        best_sd_index = np.argmax(sd_qualities)
    
    s_l = s[0][best_sd_index * (s.shape[1] // num_SDs): (best_sd_index + 1) * (s.shape[1] // num_SDs)]
    s_local = np.append(s_l, s[0][-1])  # [[hn-SDj-PDi hn-SDj-BS battery-SDj hn-PDi-BS]]

    return best_sd_index, s_local
        
#---------------------------------Initializing DDPG Agent--------------------------------------------------------------
ddpg_agent = MADDPG(LR_A, LR_C, s_dim, TAU, GAMMA, n_actions=a_dim, max_size=MEMORY_CAPACITY, batch_size=BATCH_SIZE, num_agents=num_SDs)
#-----------------------------------------------------------------------------------------------------------------------

#---------------------------------Initialize MATD3 Agent-----------------------------------------------------------------
matd3_agent = MATD3(
    alpha=LR_A_MATD3,                # Learning rate for the actor
    beta=LR_C_MATD3,                 # Learning rate for the critic
    input_dims=s_dim,                # The state dimension (s_dim)
    tau=TAU_MATD3,                   # Target network update parameter
    gamma=GAMMA_MATD3,               # Discount factor
    n_actions=a_dim,                 # Number of actions
    max_size=MEMORY_CAPACITY_MATD3,  # Size of the experience replay buffer
    batch_size=BATCH_SIZE_MATD3,     # Batch size for learning
    num_agents=num_SDs               # Total number of agents
)
#-----------------------------------------------------------------------------------------------------------------------

#---------------------------------Initialize MAPPO POLICY-----------------------------------------------------------------
mappo_agent = MAPPO(num_SDs, s_dim_ppo, a_dim_ppo, LR_A, LR_C, GAMMA, K_epochs_ppo, eps_clip_ppo, has_continuous_action_space, action_std)

#---------------------------------Initialize MASAC Agent ---------------------------------------------------------------
masac_agent = MASACAgent(
    alpha=LR_A_MASAC,
    beta=LR_C_MASAC,
    input_dims=s_dim_ppo,  # Local state dimension for each SD.
    tau=TAU_MASAC,
    gamma=GAMMA_MASAC,
    n_actions=a_dim_ppo,
    max_size=BUFFER_CAPACITY_MASAC,
    batch_size=BATCH_SIZE_MASAC,
    num_agents=num_SDs,
    reward_scale=1, # You might need to tune this
)
#-----------------------------------------------------------------------------------------------------------------------

var = 1
total_time = 1

ep_rewardall_ddpg = []
ep_rewardall_matd3 = []
ep_rewardall_mappo = []
ep_rewardall_greedy = []
ep_rewardall_random = []
ep_rewardall_masac = []

eh_rewardall_ddpg = []
eh_rewardall_matd3 = []
eh_rewardall_mappo = [] 
eh_rewardall_greedy = []
eh_rewardall_random = []
eh_rewardall_masac = []

ee_rewardall_ddpg = []
ee_rewardall_matd3 = []
ee_rewardall_mappo = [] 
ee_rewardall_greedy = []
ee_rewardall_random = []
ee_rewardall_masac = []

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
    s_matd3 = s
    s_mappo = s
    s_greedy = s
    s_random = s
    s_masac = s

    ep_ee_ddpg = 0
    ep_ee_matd3 = 0
    ep_ee_mappo = 0
    ep_ee_random = 0
    ep_ee_greedy = 0
    ep_ee_masac = 0

    ep_reward_ddpg = 0
    ep_reward_matd3 = 0
    ep_reward_mappo = 0
    ep_reward_random = 0
    ep_reward_greedy = 0
    ep_reward_masac = 0

    eh_reward_ddpg = 0
    eh_reward_matd3 = 0
    eh_reward_mappo = 0
    eh_reward_random = 0
    eh_reward_greedy = 0
    eh_reward_masac = 0

    for j in range(MAX_EP_STEPS):
        ######################## MADDPG ########################
        sd_ddpg, s_local_ddpg = choose_SD(s_ddpg)
        a_ddpg = list(ddpg_agent.choose_action(s_local_ddpg, sd_ddpg))
        a_ddpg[1] = np.clip(np.random.normal(a_ddpg[1], var), 0, 1)[0]

        r_ddpg, s_ddpg_, EHD, ee_ddpg = myenv.step(a_ddpg, s_ddpg / state_am, j+i)
        s_ddpg_ = s_ddpg_ * state_am
        ddpg_agent.remember(s_ddpg[0], a_ddpg, r_ddpg, s_ddpg_[0])
        ep_reward_ddpg += r_ddpg
        eh_reward_ddpg += EHD
        ep_ee_ddpg += ee_ddpg

        ######################## MATD3 ########################
        sd_matd3, s_local_matd3 = choose_SD(s_matd3)
        a_matd3 = list(matd3_agent.choose_action(s_local_matd3, sd_matd3))  # Choose action using MATD3 agent
        a_matd3[1] = np.clip(np.random.normal(a_matd3[1], var), 0, 1)[0]

        r_matd3, s_matd3_, EHD_matd3, ee_matd3 = myenv.step(a_matd3, s_matd3 / state_am, j+i)
        s_matd3_ = s_matd3_ * state_am
        matd3_agent.remember(s_matd3[0], a_matd3, r_matd3, s_matd3_[0])  # Store experience in MATD3 buffer
        ep_reward_matd3 += r_matd3
        eh_reward_matd3 += EHD_matd3
        ep_ee_matd3 += ee_matd3

        ######################## MAPPO ########################
        sd_mappo, s_local_mappo = choose_SD(s_mappo)   # <-- choosing agent
        s_local_mappo_tensor = torch.FloatTensor(s_local_mappo).unsqueeze(0)  # shape [1, s_dim]

        with torch.no_grad():
            a, logprob_mappo = mappo_agent.select_action(s_local_mappo_tensor, sd_mappo)

        a_mappo = [sd_mappo]
        a_mappo.append(a[0])

        # Clip the second action component
        a_mappo[1] = np.clip(np.random.normal(a_mappo[1], var), 0, 1)[0]

        # Interact with environment
        r_mappo, s_mappo_, EHD_mappo, ee_mappo = myenv.step(a_mappo, s_mappo / state_am, j+i)
        s_mappo_ = s_mappo_ * state_am

        # Save experience into MAPPO buffer
        mappo_agent.buffer.states[sd_mappo].append(s_local_mappo_tensor)
        mappo_agent.buffer.actions[sd_mappo].append(torch.FloatTensor(a_mappo))
        mappo_agent.buffer.logprobs[sd_mappo].append(logprob_mappo)
        mappo_agent.buffer.rewards[sd_mappo].append(torch.tensor(r_mappo, dtype=torch.float32))
        if i == MAX_EPISODES - 1:
            mappo_agent.buffer.is_terminals[sd_mappo].append(True)  # Set to True only at episode end
        else:
            mappo_agent.buffer.is_terminals[sd_mappo].append(False)


        ep_reward_mappo += r_mappo
        eh_reward_mappo += EHD_mappo
        ep_ee_mappo += ee_mappo

        ######################## MASAC ########################
        sd_masac, s_local_masac = choose_SD(s_masac)
        s_local_masac_tensor = torch.FloatTensor(s_local_masac).unsqueeze(0)

        with torch.no_grad():
            a_masac, _ = masac_agent.choose_action(s_local_masac_tensor, agent_index=sd_masac, evaluate=False) # Pass the agent_index

        a_masac_clipped = np.clip(a_masac, 0, 1)
        action_masac = [sd_masac, a_masac_clipped]

        r_masac, s_masac_, EHD_masac, ee_masac = myenv.step(action_masac, s_masac / state_am, j + i)
        s_masac_ = s_masac_ * state_am

        masac_agent.remember(s_masac[0], action_masac, r_masac, s_masac_[0])
        ep_reward_masac += r_masac
        eh_reward_masac += EHD_masac
        ep_ee_masac += ee_masac

        ######################## Greedy ########################
        sd_greedy, _ = choose_SD(s_greedy)
        r_greedy, s_next_greedy, EHG, ee_greedy = myenv.step_greedy(s_greedy / state_am, sd_greedy, j+i)
        s_greedy = s_next_greedy * state_am
        ep_reward_greedy += r_greedy
        eh_reward_greedy += EHG
        ep_ee_greedy += ee_greedy

        ######################## Random ########################
        sd_random, _ = choose_SD(s_random)
        r_random, s_next_random, EHR, ee_random = myenv.step_random(s_random / state_am,sd_random, j+i)
        s_random = s_next_random * state_am
        ep_reward_random += r_random
        eh_reward_random += EHR
        ep_ee_random += ee_random

        if var > 0.1:
            var *= .9998
        ddpg_agent.learn()
        matd3_agent.learn()
        masac_agent.learn() # Add the learning step for MASAC
        if total_time % update_timestep_ppo == 0:
            mappo_agent.update()
        s_ddpg = s_ddpg_
        s_matd3 = s_matd3_
        s_mappo = s_mappo_
        s_masac = s_masac_ # Update MASAC's state
        total_time += 1

        if j == MAX_EP_STEPS - 1:
            print('Avg EE for Episode:', i,
                  ' DDPG: %i' % int(ep_ee_ddpg / MAX_EP_STEPS),
                  'MATD3: %i' % int(ep_ee_matd3 / MAX_EP_STEPS),
                  'MAPPO: %i' % int(ep_ee_mappo / MAX_EP_STEPS),
                  'MASAC: %i' % int(ep_ee_masac / MAX_EP_STEPS), # Print MASAC's EE
                  'Greedy: %i' % int(ep_ee_greedy / MAX_EP_STEPS),
                  'Random: %i' % int(ep_ee_random / MAX_EP_STEPS))
                  
            
    ep_reward_ddpg = np.reshape(ep_reward_ddpg / MAX_EP_STEPS, (1,))
    ep_rewardall_ddpg.append(ep_reward_ddpg)
    eh_reward_ddpg = np.reshape(eh_reward_ddpg / MAX_EP_STEPS, (1,))
    eh_rewardall_ddpg.append(eh_reward_ddpg)
    ep_ee_ddpg = np.reshape(ep_ee_ddpg / MAX_EP_STEPS, (1,))
    ee_rewardall_ddpg.append(ep_ee_ddpg)

    ep_reward_matd3 = np.reshape(ep_reward_matd3 / MAX_EP_STEPS, (1,))
    ep_rewardall_matd3.append(ep_reward_matd3)
    eh_reward_matd3 = np.reshape(eh_reward_matd3 / MAX_EP_STEPS, (1,))
    eh_rewardall_matd3.append(eh_reward_matd3)
    ep_ee_matd3 = np.reshape(ep_ee_matd3 / MAX_EP_STEPS, (1,))
    ee_rewardall_matd3.append(ep_ee_matd3)

    ep_reward_mappo = np.reshape(ep_reward_mappo / MAX_EP_STEPS, (1,))
    ep_rewardall_mappo.append(ep_reward_mappo)
    eh_reward_mappo = np.reshape(eh_reward_mappo / MAX_EP_STEPS, (1,))
    eh_rewardall_mappo.append(eh_reward_mappo)
    ep_ee_mappo = np.reshape(ep_ee_mappo / MAX_EP_STEPS, (1,))
    ee_rewardall_mappo.append(ep_ee_mappo)

    ep_reward_masac = np.reshape(ep_reward_masac / MAX_EP_STEPS, (1,))
    ep_rewardall_masac.append(ep_reward_masac)
    eh_reward_masac = np.reshape(eh_reward_masac / MAX_EP_STEPS, (1,))
    eh_rewardall_masac.append(eh_reward_masac)
    ep_ee_masac = np.reshape(ep_ee_masac / MAX_EP_STEPS, (1,))
    ee_rewardall_masac.append(ep_ee_masac)
    
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
               sum(ep_rewardall_matd3)/len(ep_rewardall_matd3),
               sum(ep_rewardall_mappo)/len(ep_rewardall_mappo),
               sum(ep_rewardall_masac)/len(ep_rewardall_masac),
                sum(ep_rewardall_greedy)/len(ep_rewardall_greedy),
                sum(ep_rewardall_random)/len(ep_rewardall_random)]

avg_harvested_energy = [sum(eh_rewardall_ddpg)/len(eh_rewardall_ddpg),
                        sum(eh_rewardall_matd3)/len(eh_rewardall_matd3),
                        sum(eh_rewardall_mappo)/len(eh_rewardall_mappo),
                        sum(eh_rewardall_masac)/len(eh_rewardall_masac),
                        sum(eh_rewardall_greedy)/len(eh_rewardall_greedy),
                        sum(eh_rewardall_random)/len(eh_rewardall_random)]

# ########### PLOTTING FIGURES ###############
fig, ax = plt.subplots()
ax.plot(ep_rewardall_ddpg, "^-", label='MADDPG', linewidth=0.75 , color= 'darkblue')
ax.plot(ep_rewardall_matd3, "s-", label='MATD3', linewidth=0.75, color='green')
ax.plot(ep_rewardall_mappo, "d-", label='MAPPO', linewidth=0.75)
ax.plot(ep_rewardall_masac, "x-", label='MASAC', color='red', linewidth=0.75)
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
ax.plot(eh_rewardall_ddpg, "^-", label='MADDPG', linewidth=0.75 , color= 'darkblue')
ax.plot(eh_rewardall_matd3, "s-", label='MATD3', linewidth=0.75, color='green')
ax.plot(eh_rewardall_mappo, "d-", label='MAPPO', linewidth=0.75)
ax.plot(eh_rewardall_masac, "x-", label='MASAC', color='red', linewidth=0.75)
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
ax.plot(ee_rewardall_ddpg, "^-", label='MADDPG', linewidth=0.75 , color= 'darkblue')
ax.plot(ee_rewardall_matd3, "s-", label='MATD3', linewidth=0.75, color='green')
ax.plot(ee_rewardall_mappo, "d-", label='MAPPO', linewidth=0.75)
ax.plot(ee_rewardall_masac, "x-", label='MASAC', color='red', linewidth=0.75)
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