import numpy as np
import torch
from env_simple import Env_cellular as env_simple
from env_egc import Env_cellular as env_egc
from env_sc import Env_cellular as env_sc
from env_mrc import Env_cellular as env_mrc
from maddpg import MADDPG
from mappo import MAPPO
from masac import MASAC
import matplotlib.pyplot as plt

from matd3 import MATD3

######################### Hyperparameters #########################
Emax = 0.3
num_PDs = 2
num_SDs = 2
L = 2
T = 1
eta = 0.7
Pn = 1
Pmax = 0.2
w_csk = 0.000001
w_d = 1.5 * 1e-5
w_egc = 1e-6
w_mrc = 2e-6
MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001
LR_C = 0.005
GAMMA = 0.9
TAU = 0.01
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
s_dim = (3 * num_SDs) + 1
a_dim = 2
state_am = 1000

torch.manual_seed(0)
np.random.seed(0)

location_PDs = np.array([[0, 1],[0,1000]])
location_SDs = np.array([[1,1],[1,1000]])
fading_SD_BS = np.ones(num_SDs)
fading_PD_BS = np.array([0.79643, 0.87451])
hnx = np.array([[[0.90877, 0.80627], [0.82154, 0.96098]], [[0.6277, 0.63398], [0.60789, 0.92472]]])

fading_PD_SD = np.zeros((num_PDs, num_SDs))

###################### Select diversity mode ######################
diversity_mode = int(input("Enter diversity technique (0: Simple, 1: EGC, 2: MRC, 3: SC): "))
if diversity_mode == 1:
    for i in range(num_PDs):
        for j in range(num_SDs):
            fading_PD_SD[i][j] = np.sum(hnx[i][j]) ** 2
    env = env_egc(MAX_EP_STEPS, s_dim, location_PDs, location_SDs, Emax, num_PDs, L, T, eta, Pn, Pmax, w_d, w_egc, w_csk, fading_PD_SD, fading_PD_BS, fading_SD_BS, num_SDs)
elif diversity_mode == 2:
    for i in range(num_PDs):
        for j in range(num_SDs):
            fading_PD_SD[i][j] = np.sum(hnx[i][j] ** 2)
    env = env_mrc(MAX_EP_STEPS, s_dim, location_PDs, location_SDs, Emax, num_PDs, T, eta, Pn, Pmax, w_d, w_mrc, w_csk, fading_PD_SD, fading_PD_BS, fading_SD_BS, num_SDs)
elif diversity_mode == 3:
    for i in range(num_PDs):
        for j in range(num_SDs):
            fading_PD_SD[i][j] = np.max(hnx[i][j] ** 2)
    env = env_sc(MAX_EP_STEPS, s_dim, location_PDs, location_SDs, Emax, num_PDs, T, eta, Pn, Pmax, w_d, w_csk, fading_PD_SD, fading_PD_BS, fading_SD_BS, num_SDs)
else:
    for i in range(num_PDs):
        for j in range(num_SDs):
            fading_PD_SD[i][j] = np.random.choice(hnx[i][j])
    env = env_simple(MAX_EP_STEPS, s_dim, location_PDs, location_SDs, Emax, num_PDs, T, eta, Pn, Pmax, w_csk, fading_PD_SD, fading_PD_BS, fading_SD_BS, num_SDs)

#---------------------------------Initializing DDPG Agent--------------------------------------------------------------
ddpg_agent = MADDPG(LR_A, LR_C, s_dim, TAU, GAMMA, n_actions=a_dim, max_size=MEMORY_CAPACITY, batch_size=BATCH_SIZE, num_agents=num_SDs)

#---------------------------------Initialize MATD3 Agent-----------------------------------------------------------------
matd3_agent = MATD3(
    alpha=LR_A,                # Learning rate for the actor
    beta=LR_C,                 # Learning rate for the critic
    input_dims=s_dim,                # The state dimension (s_dim)
    tau=TAU,                   # Target network update parameter
    gamma=GAMMA,               # Discount factor
    n_actions=a_dim,                 # Number of actions
    max_size=MEMORY_CAPACITY,  # Size of the experience replay buffer
    batch_size=BATCH_SIZE,     # Batch size for learning
    num_agents=num_SDs               # Total number of agents
)
#---------------------------------Initialize MAPPO POLICY-----------------------------------------------------------------
mappo_agent = MAPPO(num_agents=num_SDs, local_state_dim=s_dim//num_SDs+1, global_state_dim=s_dim, action_dim_per_agent=1, lr_actor=LR_A, lr_critic=LR_C, gamma=GAMMA, K_epochs=4, eps_clip=0.2, buffer_size=1000, has_continuous_action_space=True, action_std_init=0.6)
#-------------------------------------Initialize MASAC-------------------------------------------------------------------
masac_agent = MASAC(LR_A, LR_C, s_dim, TAU, GAMMA, a_dim, MEMORY_CAPACITY, BATCH_SIZE, num_SDs)

def choose_SD(state):
    sd_qualities = []
    for sd in range(num_SDs):
        sd_state = state[0][sd * 3: (sd + 1) * 3]
        norm = (sd_state - np.min(sd_state)) / (np.max(sd_state) - np.min(sd_state) + 1e-8)
        quality = np.sum(norm)
        sd_qualities.append(quality)

    if np.random.rand() < 0.1:  # 10% chance to explore
        return np.random.randint(num_SDs)
    return np.argmax(sd_qualities)


# === Storage for EEs ===
ee_rewardall_ddpg = []
ee_rewardall_matd3 = []
ee_rewardall_masac = []
ee_rewardall_mappo = []
ee_rewardall_greedy = []
ee_rewardall_random = []

# === Training Loop ===
var = 1.0
for i in range(MAX_EPISODES):
    battery = env.reset()
    s = []
    for sd in range(num_SDs):
        s.extend([env.hn_PD_SD[i % num_PDs, sd], env.hn_SD_BS[sd], battery[sd]])
    s.append(env.hn_PD_BS[i % num_PDs])
    s = np.reshape(s, (1, s_dim)) * state_am

    s_ddpg = s.copy()
    s_matd3 = s.copy()
    s_masac = s.copy()
    s_mappo = s.copy()
    s_greedy = s.copy()
    s_random = s.copy()

    ee_ddpg = ee_matd3 = ee_masac = ee_mappo = ee_greedy = ee_random = 0

    for j in range(MAX_EP_STEPS):
        def act_and_step(agent, s_input, agent_name):
            sd = choose_SD(s_input)
            s_local = np.append(s_input[0][sd * 3:(sd + 1) * 3], s_input[0][-1])
            a = agent.choose_action(s_local, sd)
            a = np.clip(np.random.normal(a, var), 0, 1)[0]
            action_vec = np.zeros(a_dim, dtype=np.float32)
            action_vec[sd] = a
            r, s_next, _, ee, _ = env.step(sd, a, s_input / state_am, j+i)
            s_next = s_next * state_am
            agent.remember(s_input[0], action_vec, r, s_next[0])
            agent.learn()
            return s_next, ee

        s_ddpg, ee1 = act_and_step(ddpg_agent, s_ddpg, "ddpg")
        ee_ddpg += ee1

        s_matd3, ee2 = act_and_step(matd3_agent, s_matd3, "matd3")
        ee_matd3 += ee2

        s_masac, ee3 = act_and_step(masac_agent, s_masac, "masac")
        ee_masac += ee3

        # MAPPO specific
        sd = choose_SD(s_mappo)
        s_local = np.append(s_mappo[0][sd * 3:(sd + 1) * 3], s_mappo[0][-1])
        a_mappo, logprob = mappo_agent.select_action(s_local, sd)
        a_mappo = np.clip(np.random.normal(a_mappo, var), 0, 1)[0]
        action_vec = np.zeros(a_dim, dtype=np.float32)
        action_vec[sd] = a_mappo
        r, s_next, _, _, ee = env.step(sd, a_mappo, s_mappo / state_am, j+i)
        s_next = s_next * state_am
        mappo_agent.store_transition(s_mappo, action_vec, logprob, r, (j == MAX_EP_STEPS-1), sd)
        if mappo_agent.buffer.ptr == mappo_agent.buffer_size:
            mappo_agent.update()
            mappo_agent.buffer.clear()
        s_mappo = s_next
        ee_mappo += ee

        # Greedy
        sd = choose_SD(s_greedy)
        _, s_greedy_next, _, _, ee = env.step_greedy(s_greedy / state_am, sd, j+i)
        s_greedy = s_greedy_next * state_am
        ee_greedy += ee

        # Random
        sd = choose_SD(s_random)
        _, s_random_next, _, _, ee = env.step_random(s_random / state_am, sd, j+i)
        s_random = s_random_next * state_am
        ee_random += ee

        var = max(var * 0.9998, 0.1)

    # Store average EE
    ee_rewardall_ddpg.append(ee_ddpg / MAX_EP_STEPS)
    ee_rewardall_matd3.append(ee_matd3 / MAX_EP_STEPS)
    ee_rewardall_masac.append(ee_masac / MAX_EP_STEPS)
    ee_rewardall_mappo.append(ee_mappo / MAX_EP_STEPS)
    ee_rewardall_greedy.append(ee_greedy / MAX_EP_STEPS)
    ee_rewardall_random.append(ee_random / MAX_EP_STEPS)

    print(f"[Episode {i}] EE -> DDPG: {ee_ddpg/MAX_EP_STEPS:.4f}, MATD3: {ee_matd3/MAX_EP_STEPS:.4f}, MASAC: {ee_masac/MAX_EP_STEPS:.4f}, MAPPO: {ee_mappo/MAX_EP_STEPS:.4f}, Greedy: {ee_greedy/MAX_EP_STEPS:.4f}, Random: {ee_random/MAX_EP_STEPS:.4f}")
MAX_EPISODES = len(ee_rewardall_ddpg)  # Ensure all lists are of same length

# ------------------- Plot: Average Sum Rate -------------------
fig, ax = plt.subplots()
ax.plot(ee_rewardall_ddpg, "^-", label='MADDPG', linewidth=0.75 , color='darkblue')
ax.plot(ee_rewardall_matd3, "s-", label='MATD3', linewidth=0.75, color='green')
ax.plot(ee_rewardall_masac, "v-", label='MASAC', linewidth=0.75, color='skyblue')
ax.plot(ee_rewardall_mappo, "d-", label='MAPPO', linewidth=0.75, color='red')
ax.plot(ee_rewardall_greedy, "o-", label='Greedy', linewidth=0.75, color='orange')
ax.plot(ee_rewardall_random, "x-", label='Random', linewidth=0.75, color='black')

ax.set_xlabel("Episodes")
ax.set_ylabel("Average Sum Rate (b/s)")
ax.margins(x=0)
ax.set_xlim(0, MAX_EPISODES)
ax.grid(which="both", axis='y', linestyle=':', color='lightgray', linewidth=0.5)
ax.minorticks_on()
ax.tick_params(which="minor", bottom=False, left=False)

legend = ax.legend(facecolor='none')
legend.set_draggable(True)

# Save plots
fig.savefig("avg_sumrate.eps", format="eps", bbox_inches="tight")
fig.savefig("avg_sumrate.png", format="png", bbox_inches="tight")

plt.show()