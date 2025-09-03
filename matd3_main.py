import numpy as np
import torch
import matplotlib.pyplot as plt
from matd3 import MATD3
from env_simple import Env_cellular as env_simple
from env_egc import Env_cellular as env_egc
from env_sc import Env_cellular as env_sc
from env_mrc import Env_cellular as env_mrc

# === Hyperparameters ===
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

# Best parameters from tuning results
LR_A = 0.00013575014643156606
LR_C = 0.008801517965274138
GAMMA = 0.8500000000000001
TAU = 0.035603917735483624
MEMORY_CAPACITY = 13000
BATCH_SIZE = 128

s_dim = (3 * num_SDs) + 1
a_dim = num_SDs
state_am = 1000

np.random.seed(0)
torch.manual_seed(0)

# === Diversity Setup ===
location_PDs = np.array([[0, 1], [0, 1000]])
location_SDs = np.array([[1, 1], [1, 1000]])
fading_SD_BS = np.ones(num_SDs)
fading_PD_BS = np.array([0.79643, 0.87451])
hnx = np.array([[[0.90877, 0.80627], [0.82154, 0.96098]],
                [[0.6277, 0.63398], [0.60789, 0.92472]]])
fading_PD_SD = np.zeros((num_PDs, num_SDs))

diversity_mode = int(input("Enter diversity technique (0: Simple, 1: EGC, 2: MRC, 3: SC): "))
if diversity_mode == 1:
    for i in range(num_PDs):
        for j in range(num_SDs):
            fading_PD_SD[i][j] = np.sum(hnx[i][j]) ** 2
    env = env_egc(MAX_EP_STEPS, s_dim, location_PDs, location_SDs, Emax, num_PDs, L, T, eta, Pn, Pmax,
                  w_d, w_egc, w_csk, fading_PD_SD, fading_PD_BS, fading_SD_BS, num_SDs)
elif diversity_mode == 2:
    for i in range(num_PDs):
        for j in range(num_SDs):
            fading_PD_SD[i][j] = np.sum(hnx[i][j] ** 2)
    env = env_mrc(MAX_EP_STEPS, s_dim, location_PDs, location_SDs, Emax, num_PDs, T, eta, Pn, Pmax,
                  w_d, w_mrc, w_csk, fading_PD_SD, fading_PD_BS, fading_SD_BS, num_SDs)
elif diversity_mode == 3:
    for i in range(num_PDs):
        for j in range(num_SDs):
            fading_PD_SD[i][j] = np.max(hnx[i][j] ** 2)
    env = env_sc(MAX_EP_STEPS, s_dim, location_PDs, location_SDs, Emax, num_PDs, T, eta, Pn, Pmax,
                 w_d, w_csk, fading_PD_SD, fading_PD_BS, fading_SD_BS, num_SDs)
else:
    for i in range(num_PDs):
        for j in range(num_SDs):
            fading_PD_SD[i][j] = np.random.choice(hnx[i][j])
    env = env_simple(MAX_EP_STEPS, s_dim, location_PDs, location_SDs, Emax, num_PDs, T, eta, Pn, Pmax,
                     w_csk, fading_PD_SD, fading_PD_BS, fading_SD_BS, num_SDs)

# === Agent Init ===
# Best parameters from tuning results
matd3_agent = MATD3(s_dim, a_dim, num_SDs, lr_actor=LR_A, lr_critic=LR_C,
                    gamma=GAMMA, tau=TAU, max_size=MEMORY_CAPACITY, batch_size=BATCH_SIZE,
                    policy_noise=0.1, noise_clip=0.5, policy_delay=5)

# === Device selection ===
def choose_SD(state):
    scores = []
    for sd in range(num_SDs):
        norm = state[0][sd * 3:(sd + 1) * 3]
        scores.append(np.sum(norm / (np.max(norm) + 1e-8)))
    return np.random.randint(num_SDs) if np.random.rand() < 0.1 else np.argmax(scores)

# === Training ===
dr_rewardall_td3 = []
ee_rewardall_td3 = []
eh_rewardall_td3 = []

# Lists to store SD selection frequencies for each episode
sd0_selections = []
sd1_selections = []

for ep in range(MAX_EPISODES):
    battery = env.reset()
    s = []
    for sd in range(num_SDs):
        s.extend([env.hn_PD_SD[ep % num_PDs, sd], env.hn_SD_BS[sd], battery[sd]])
    s.append(env.hn_PD_BS[ep % num_PDs])
    s = np.reshape(s, (1, s_dim)) * state_am
    s_td3 = s.copy()
    sr, ee, eh = 0, 0, 0
    
    # Counters for this episode
    sd0_count = 0
    sd1_count = 0

    for step in range(MAX_EP_STEPS):
        actions = np.zeros(a_dim)
        for agent_idx in range(num_SDs):
            s_local = np.append(s_td3[0][agent_idx * 3:(agent_idx + 1) * 3], s_td3[0][-1])
            a = matd3_agent.choose_action(s_local, agent_idx, explore=True)
            actions[agent_idx] = np.clip(a[0], 0, 1)

        sd = choose_SD(s_td3)
        
        # Count SD selections
        if sd == 0:
            sd0_count += 1
        elif sd == 1:
            sd1_count += 1
            
        r, s_next, ehp, eep, srp = env.step(sd, actions[sd], s_td3 / state_am, step + ep)
        s_next *= state_am

        matd3_agent.remember(s_td3.flatten(), actions.copy(), r, s_next.flatten())
        matd3_agent.learn()

        s_td3 = s_next
        sr += srp
        ee += eep
        eh += ehp

    # Store the counts for this episode
    sd0_selections.append(sd0_count)
    sd1_selections.append(sd1_count)
    
    dr_rewardall_td3.append(sr / MAX_EP_STEPS)
    ee_rewardall_td3.append(ee / MAX_EP_STEPS)
    eh_rewardall_td3.append(eh)
    print(f"[Episode {ep}] SR -> MATD3: {sr/MAX_EP_STEPS:.4f} - EE -> MATD3: {ee/MAX_EP_STEPS:.4f} - EH -> MATD3: {eh:.4f}")
    print(f"  SD Counts: SD0={sd0_count}, SD1={sd1_count}")
print("matd3")
print(f"SUMRATE-> {dr_rewardall_td3}")
print("----------------------------------")
print(f"EE-> {ee_rewardall_td3}")
print("----------------------------------")
print(f"EH-> {eh_rewardall_td3}")
print("----------------------------------")

# Print SD selection frequencies
print("SD Selection Frequencies:")
print(f"SD 0 selections per episode: {sd0_selections}")
print(f"SD 1 selections per episode: {sd1_selections}")
print("----------------------------------")
# === Plotting ===
fig, ax = plt.subplots()
ax.plot(dr_rewardall_td3, "d-", label='MATD3', linewidth=0.75, color='green')
ax.set_xlabel("Episodes")
ax.set_ylabel("Average Sum Rate (b/s)")
ax.margins(x=0)
ax.set_xlim(0, MAX_EPISODES)
ax.grid(which="both", axis='y', linestyle=':', color='lightgray', linewidth=0.5)
ax.minorticks_on()
ax.tick_params(which="minor", bottom=False, left=False)
ax.legend(facecolor='none').set_draggable(True)
fig.savefig(f"matd3_avg_sumrate_{diversity_mode}.eps", format="eps", bbox_inches="tight")
fig.savefig(f"matd3_avg_sumrate_{diversity_mode}.png", format="png", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
ax.plot(ee_rewardall_td3, "d-", label='MATD3', linewidth=0.75, color='purple')
ax.set_xlabel("Episodes")
ax.set_ylabel("Average Energy Efficiency")
ax.margins(x=0)
ax.set_xlim(0, MAX_EPISODES)
ax.grid(which="both", axis='y', linestyle=':', color='lightgray', linewidth=0.5)
ax.minorticks_on()
ax.tick_params(which="minor", bottom=False, left=False)
ax.legend(facecolor='none').set_draggable(True)
fig.savefig(f"matd3_avg_ee_{diversity_mode}.eps", format="eps", bbox_inches="tight")
fig.savefig(f"matd3_avg_ee_{diversity_mode}.png", format="png", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
ax.plot(eh_rewardall_td3, "s-", label='MATD3', linewidth=0.75, color='green')
ax.set_xlabel("Episodes")
ax.set_ylabel("Energy Harvested")
ax.margins(x=0)
ax.set_xlim(0, MAX_EPISODES)
ax.grid(which="both", axis='y', linestyle=':', color='lightgray', linewidth=0.5)
ax.minorticks_on()
ax.tick_params(which="minor", bottom=False, left=False)
ax.legend(facecolor='none').set_draggable(True)

fig.savefig(f"matd3_avg_eh_{diversity_mode}.eps", format="eps", bbox_inches="tight")
fig.savefig(f"matd3_avg_eh_{diversity_mode}.png", format="png", bbox_inches="tight")
plt.show()
