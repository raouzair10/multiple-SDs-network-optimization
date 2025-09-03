
import numpy as np
import torch
import matplotlib.pyplot as plt
from mappo import MAPPO
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
MEMORY_CAPACITY = 10000
state_am = 1000

s_dim = (3 * num_SDs) + 1
a_dim = num_SDs

np.random.seed(0)
torch.manual_seed(0)

# === Topology and Channel ===
location_PDs = np.array([[0, 1], [0, 1000]])
location_SDs = np.array([[1, 1], [1, 1000]])
fading_SD_BS = np.ones(num_SDs)
fading_PD_BS = np.array([0.79643, 0.87451])
hnx = np.array([[[0.90877, 0.80627], [0.82154, 0.96098]],
                [[0.6277, 0.63398], [0.60789, 0.92472]]])
fading_PD_SD = np.zeros((num_PDs, num_SDs))

# === Diversity Setup ===
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
mappo_agent = MAPPO(num_SDs, 4, s_dim, 1, 
                     action_std_init=0.2, gamma=0.92, eps_clip=0.5, K_epochs=3,
                     buffer_size=2000, lr_actor=0.006468776041288534, lr_critic=0.005462588876142524)

# === Device selection ===
def choose_SD(state):
    scores = []
    for sd in range(num_SDs):
        norm = state[0][sd * 3:(sd + 1) * 3]
        scores.append(np.sum(norm / (np.max(norm) + 1e-8)))
    return np.random.randint(num_SDs) if np.random.rand() < 0.1 else np.argmax(scores)

dr_rewardall = []
ee_rewardall = []
eh_rewardall = []

# Lists to store SD selection frequencies for each episode
sd0_selections = []
sd1_selections = []

# === Training ===
for ep in range(MAX_EPISODES):
    battery = env.reset()
    s = []
    for sd in range(num_SDs):
        s.extend([env.hn_PD_SD[ep % num_PDs, sd], env.hn_SD_BS[sd], battery[sd]])
    s.append(env.hn_PD_BS[ep % num_PDs])
    s = np.reshape(s, (1, s_dim)) * state_am
    s_mappo = s.copy()
    sr, ee, eh = 0, 0, 0
    
    # Counters for this episode
    sd0_count = 0
    sd1_count = 0

    for step in range(MAX_EP_STEPS):
        actions = np.zeros(a_dim)
        logprobs = np.zeros(a_dim)

        for agent_idx in range(num_SDs):
            local_state = np.append(s_mappo[0][agent_idx * 3:(agent_idx + 1) * 3], s_mappo[0][-1])
            a, logp = mappo_agent.select_action(local_state, agent_idx)
            actions[agent_idx] = np.clip(a[0], 0, 1)
            logprobs[agent_idx] = logp

        sd = choose_SD(s_mappo)
        
        # Count SD selections
        if sd == 0:
            sd0_count += 1
        elif sd == 1:
            sd1_count += 1
            
        r, s_next, ehp, eep, srp = env.step(sd, actions[sd], s_mappo / state_am, step + ep)
        s_next *= state_am

        mappo_agent.store_transition(s_mappo.flatten(), actions.copy(), logprobs[sd], r, False, sd)
        mappo_agent.update()

        s_mappo = s_next
        sr += srp
        ee += eep
        eh += ehp

    # Store the counts for this episode
    sd0_selections.append(sd0_count)
    sd1_selections.append(sd1_count)
    
    dr_rewardall.append(sr / MAX_EP_STEPS)
    ee_rewardall.append(ee / MAX_EP_STEPS)
    eh_rewardall.append(eh)
    print(f"[Episode {ep}] SR -> MAPPO: {sr/MAX_EP_STEPS:.4f} - EE -> MAPPO: {ee/MAX_EP_STEPS:.4f} - EH -> MAPPO: {eh:.4f}")
    print(f"  SD Counts: SD0={sd0_count}, SD1={sd1_count}")
print("mappo")
print(f"SUMRATE-> {dr_rewardall}")
print("----------------------------------")
print(f"EE-> {ee_rewardall}")
print("----------------------------------")
print(f"EH-> {eh_rewardall}")
print("----------------------------------")

# Print SD selection frequencies
print("SD Selection Frequencies:")
print(f"SD 0 selections per episode: {sd0_selections}")
print(f"SD 1 selections per episode: {sd1_selections}")
print("----------------------------------")
# === Plotting ===
fig, ax = plt.subplots()
ax.plot(dr_rewardall, "d-", label='MAPPO', linewidth=0.75, color='blue')
ax.set_xlabel("Episodes")
ax.set_ylabel("Average Sum Rate (b/s)")
ax.set_xlim(0, MAX_EPISODES)
ax.grid(which="both", axis='y', linestyle=':', color='lightgray', linewidth=0.5)
ax.minorticks_on()
ax.tick_params(which="minor", bottom=False, left=False)
ax.legend(facecolor='none').set_draggable(True)
fig.savefig(f"mappo_avg_sumrate_{diversity_mode}.png", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
ax.plot(ee_rewardall, "d-", label='MAPPO', linewidth=0.75, color='orange')
ax.set_xlabel("Episodes")
ax.set_ylabel("Average Energy Efficiency")
ax.set_xlim(0, MAX_EPISODES)
ax.grid(which="both", axis='y', linestyle=':', color='lightgray', linewidth=0.5)
ax.minorticks_on()
ax.tick_params(which="minor", bottom=False, left=False)
ax.legend(facecolor='none').set_draggable(True)
fig.savefig(f"mappo_avg_ee_{diversity_mode}.png", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
ax.plot(eh_rewardall, "s-", label='MAPPO', linewidth=0.75, color='green')
ax.set_xlabel("Episodes")
ax.set_ylabel("Energy Harvested")
ax.margins(x=0)
ax.set_xlim(0, MAX_EPISODES)
ax.grid(which="both", axis='y', linestyle=':', color='lightgray', linewidth=0.5)
ax.minorticks_on()
ax.tick_params(which="minor", bottom=False, left=False)
ax.legend(facecolor='none').set_draggable(True)
fig.savefig(f"mappo_avg_eh_{diversity_mode}.png", bbox_inches="tight")
plt.show()
