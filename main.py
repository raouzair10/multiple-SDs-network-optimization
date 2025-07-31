import numpy as np
import torch
import random
from env_simple import Env_cellular as env_simple
from env_egc import Env_cellular as env_egc
from env_sc import Env_cellular as env_sc
from env_mrc import Env_cellular as env_mrc
import matplotlib.pyplot as plt
import matplotlib
from masac import MASAC
from maddpg import MADDPG
from matd3 import MATD3
from mappo import MAPPO
import warnings;
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matplotlib.use('TkAgg')
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
w_csk = 0.000001
w_d = 1.5 * (10 ** -5)
w_egc = 1 * (10 ** -6)
w_mrc = 2 * (10 ** -6)

##################### Hyper Parameters #####################
MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001
LR_C = 0.005
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
location_PDs = np.array([[0, 1],[0,1000]])
location_SDs = np.array([[1,1],[1,1000]])

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
    env = env_egc(MAX_EP_STEPS, s_dim, location_PDs, location_SDs, Emax, num_PDs, L, T, eta, Pn, Pmax, w_d, w_egc, w_csk, fading_PD_SD, fading_PD_BS, fading_SD_BS, num_SDs)

elif diversity_mode == 2:   # MRC
    for i in range(num_PDs):
        for j in range(num_SDs):
            fading_PD_SD[i][j] = np.sum(hnx[i][j] ** 2) # sum of squares
    env = env_mrc(MAX_EP_STEPS, s_dim, location_PDs, location_SDs, Emax, num_PDs, T, eta, Pn, Pmax, w_d, w_mrc, w_csk, fading_PD_SD, fading_PD_BS, fading_SD_BS, num_SDs)

elif diversity_mode == 3:   # SC
    for i in range(num_PDs):
        for j in range(num_SDs):
            fading_PD_SD[i][j] = np.max(hnx[i][j] ** 2) # max squared value
    env = env_sc(MAX_EP_STEPS, s_dim, location_PDs, location_SDs, Emax, num_PDs, T, eta, Pn, Pmax, w_d, w_csk, fading_PD_SD, fading_PD_BS, fading_SD_BS, num_SDs)


else:   # simple (no diversity) 
    for i in range(num_PDs):
        for j in range(num_SDs):
            fading_PD_SD[i][j] = random.choice(hnx[i][j]) # random value
            # [[0.90877 0.96098]
            #  [0.63398 0.60789]]
            # [[PD1-SD1 PD1-SD2]
            #  [PD2-SD1 PD2SD2]]
    env = env_simple(MAX_EP_STEPS, s_dim, location_PDs, location_SDs, Emax, num_PDs, T, eta, Pn, Pmax, w_csk, fading_PD_SD, fading_PD_BS, fading_SD_BS, num_SDs)

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

    return best_sd_index
        
#---------------------------------Initializing DDPG Agent--------------------------------------------------------------

maddpg_agent = MADDPG(s_dim, a_dim, num_SDs, lr_actor=LR_A, lr_critic=LR_C,
                 gamma=GAMMA, tau=TAU, max_size=MEMORY_CAPACITY, batch_size=BATCH_SIZE)
#-----------------------------------------------------------------------------------------------------------------------

#---------------------------------Initialize MATD3 Agent-----------------------------------------------------------------
matd3_agent = MATD3(s_dim, a_dim, num_SDs, lr_actor=LR_A, lr_critic=LR_C,
                    gamma=GAMMA, tau=TAU, max_size=5000, batch_size=BATCH_SIZE)
#-----------------------------------------------------------------------------------------------------------------------

#---------------------------------Initialize MAPPO Agent----------------------------------------------------------------
mappo_agent = MAPPO(num_agents=num_SDs, local_state_dim=4, global_state_dim=s_dim, action_dim=1)
#-----------------------------------------------------------------------------------------------------------------------

#---------------------------------Initialize MASAC Agent-----------------------------------------------------------------
masac_agent = MASAC(lr_a=LR_A, lr_c=LR_C, global_input_dims=s_dim, tau=TAU, gamma=GAMMA, n_actions=a_dim, max_size=MEMORY_CAPACITY, batch_size=BATCH_SIZE, num_agents=num_SDs)
#-----------------------------------------------------------------------------------------------------------------------

# ========== Training ========== #
dr_rewardall_maddpg, ee_rewardall_maddpg = [], []
dr_rewardall_mappo, ee_rewardall_mappo = [], []
dr_rewardall_td3, ee_rewardall_td3 = [], []
dr_rewardall_masac, ee_rewardall_masac = [], []

var = 1.0

for ep in range(MAX_EPISODES):
    battery = env.reset()
    s = []
    for sd in range(num_SDs):
        s.extend([env.hn_PD_SD[ep % num_PDs, sd], env.hn_SD_BS[sd], battery[sd]])
    s.append(env.hn_PD_BS[ep % num_PDs])
    s = np.reshape(s, (1, s_dim)) * state_am

    s_maddpg = s.copy()
    s_mappo = s.copy()
    s_td3 = s.copy()
    s_masac = s.copy()

    sr_maddpg = sr_mappo = sr_td3 = sr_masac = 0
    ee_maddpg = ee_mappo = ee_td3 = ee_masac = 0

    for step in range(MAX_EP_STEPS):
        # --- MADDPG ---
        sd = choose_SD(s_maddpg)
        s_local = np.append(s_maddpg[0][sd * 3:(sd + 1) * 3], s_maddpg[0][-1])
        a = maddpg_agent.choose_action(s_local, sd)
        a = np.clip(np.random.normal(a, var), 0, 1)[0]
        action_vec = np.zeros(a_dim, dtype=np.float32)
        action_vec[sd] = a
        r, s_next, _, eep, srp = env.step(sd, a, s_maddpg / state_am, step + ep)
        s_next *= state_am
        maddpg_agent.remember(s_maddpg[0], action_vec, r, s_next[0])
        maddpg_agent.learn()
        s_maddpg = s_next
        sr_maddpg += srp
        ee_maddpg += eep

        # --- MAPPO ---
        actions = np.zeros(a_dim)
        logprobs = np.zeros(a_dim)
        for agent_idx in range(num_SDs):
            local_state = np.append(s_mappo[0][agent_idx * 3:(agent_idx + 1) * 3], s_mappo[0][-1])
            a, logp = mappo_agent.select_action(local_state, agent_idx)
            actions[agent_idx] = np.clip(a[0], 0, 1)
            logprobs[agent_idx] = logp
        sd = choose_SD(s_mappo)
        r, s_next, done, eep, srp = env.step(sd, actions[sd], s_mappo / state_am, step + ep)
        s_next *= state_am
        mappo_agent.store_transition(s_mappo.flatten(), actions.copy(), logprobs[sd], r, done, sd)
        mappo_agent.update()
        s_mappo = s_next
        sr_mappo += srp
        ee_mappo += eep

        # --- MATD3 ---
        actions = np.zeros(a_dim)
        for agent_idx in range(num_SDs):
            s_local = np.append(s_td3[0][agent_idx * 3:(agent_idx + 1) * 3], s_td3[0][-1])
            a = matd3_agent.choose_action(s_local, agent_idx, explore=True)
            actions[agent_idx] = np.clip(a[0], 0, 1)
        sd = choose_SD(s_td3)
        r, s_next, _, eep, srp = env.step(sd, actions[sd], s_td3 / state_am, step + ep)
        s_next *= state_am
        matd3_agent.remember(s_td3.flatten(), actions.copy(), r, s_next.flatten())
        matd3_agent.learn()
        s_td3 = s_next
        sr_td3 += srp
        ee_td3 += eep

        # --- MASAC ---
        sd = choose_SD(s_masac)
        s_local = np.append(s_masac[0][sd * 3:(sd + 1) * 3], s_masac[0][-1])
        a = masac_agent.choose_action(s_local, sd)
        a = np.clip(np.random.normal(a, var), 0, 1)[0]
        action_vec = np.zeros(a_dim, dtype=np.float32)
        action_vec[sd] = a
        r, s_next, _, eep, srp = env.step(sd, a, s_masac / state_am, step + ep)
        s_next *= state_am
        masac_agent.remember(s_masac[0], action_vec, r, s_next[0])
        masac_agent.learn()
        s_masac = s_next
        sr_masac += srp
        ee_masac += eep

    var = max(var * 0.9998, 0.1)

    dr_rewardall_maddpg.append(sr_maddpg / MAX_EP_STEPS)
    ee_rewardall_maddpg.append(ee_maddpg / MAX_EP_STEPS)
    dr_rewardall_mappo.append(sr_mappo / MAX_EP_STEPS)
    ee_rewardall_mappo.append(ee_mappo / MAX_EP_STEPS)
    dr_rewardall_td3.append(sr_td3 / MAX_EP_STEPS)
    ee_rewardall_td3.append(ee_td3 / MAX_EP_STEPS)
    dr_rewardall_masac.append(sr_masac / MAX_EP_STEPS)
    ee_rewardall_masac.append(ee_masac / MAX_EP_STEPS)

    print(f"[Episode {ep}] SR -> MADDPG: {sr_maddpg/MAX_EP_STEPS:.4f}, MAPPO: {sr_mappo/MAX_EP_STEPS:.4f}, MATD3: {sr_td3/MAX_EP_STEPS:.4f}, MASAC: {sr_masac/MAX_EP_STEPS:.4f}")

# ========== Plotting ========== #
def plot_results(rewards_dict, ylabel, filename):
    plt.figure()
    for label, rewards in rewards_dict.items():
        plt.plot(rewards, label=label)
    plt.xlabel("Episodes")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

plot_results({
    'MADDPG': dr_rewardall_maddpg,
    'MAPPO': dr_rewardall_mappo,
    'MATD3': dr_rewardall_td3,
    'MASAC': dr_rewardall_masac
}, "Average Sum Rate (b/s)", "avg_sumrate_all.png")

plot_results({
    'MADDPG': ee_rewardall_maddpg,
    'MAPPO': ee_rewardall_mappo,
    'MATD3': ee_rewardall_td3,
    'MASAC': ee_rewardall_masac
}, "Average Energy Efficiency", "avg_ee_all.png")
