import numpy as np
import random
import matplotlib.pyplot as plt
from env_simple import Env_cellular as env_simple
from env_egc import Env_cellular as env_egc
from env_sc import Env_cellular as env_sc
from env_mrc import Env_cellular as env_mrc

##################### System Parameters ####################
Emax = 0.3
num_PDs = 2
num_SDs = 2
L = 3
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
s_dim = (3 * num_SDs) + 1
state_am = 1000

##################### Setup ####################
location_PDs = np.array([[0, 1], [0, 1000]])
location_SDs = np.array([[1, 1], [1, 1000]])

fading_PD_BS = np.array([0.79643, 0.87451])
fading_SD_BS = np.ones(num_SDs)
hnx = np.array([[[0.90877, 0.80627],
                 [0.82154, 0.96098]],
                [[0.6277, 0.63398],
                 [0.60789, 0.92472]]])
fading_PD_SD = np.zeros((num_PDs, num_SDs))

diversity_mode = int(input("Enter diversity mode (0:Simple, 1:EGC, 2:MRC, 3:SC): "))

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
            fading_PD_SD[i][j] = random.choice(hnx[i][j])
    env = env_simple(MAX_EP_STEPS, s_dim, location_PDs, location_SDs, Emax, num_PDs, T, eta, Pn, Pmax, w_csk, fading_PD_SD, fading_PD_BS, fading_SD_BS, num_SDs)

def choose_SD(s):
    sd_qualities = []
    for sd_index in range(num_SDs):
        sd = np.array([s[0][sd_index * 3],
                       s[0][sd_index * 3 + 1],
                       s[0][sd_index * 3 + 2]])
        sd = (sd - sd.min()) / (sd.max() - sd.min() + 1e-8)
        sd_qualities.append(np.sum(sd))
    if np.random.rand() < 0.1:
        best_sd_index = np.random.randint(num_SDs)
    else:
        best_sd_index = np.argmax(sd_qualities)
    s_local = np.append(s[0][best_sd_index * 3: (best_sd_index + 1) * 3], s[0][-1])
    return best_sd_index, s_local

eh_rewardall_greedy, eh_rewardall_random = [], []
ee_rewardall_greedy, ee_rewardall_random = [], []
dr_rewardall_greedy, dr_rewardall_random = [], []

# Lists to store SD selection frequencies for each episode
sd0_selections_greedy = []
sd1_selections_greedy = []
sd0_selections_random = []
sd1_selections_random = []

for i in range(MAX_EPISODES):
    s = []
    battery_init = env.reset()
    for sd in range(num_SDs):
        s += [env.hn_PD_SD[i % env.num_PDs, sd], env.hn_SD_BS[sd], battery_init[sd]]
    s.append(env.hn_PD_BS[i % env.num_PDs])
    s = np.reshape(s, (1, s_dim)) * state_am
    s_greedy, s_random = s.copy(), s.copy()

    ep_eh_greedy = ep_eh_random = 0
    ep_ee_greedy = ep_ee_random = 0
    ep_dr_greedy = ep_dr_random = 0
    
    # Counters for this episode
    sd0_count_greedy = 0
    sd1_count_greedy = 0
    sd0_count_random = 0
    sd1_count_random = 0

    for j in range(MAX_EP_STEPS):
        sd_greedy, _ = choose_SD(s_greedy)
        
        # Count SD selections for greedy
        if sd_greedy == 0:
            sd0_count_greedy += 1
        elif sd_greedy == 1:
            sd1_count_greedy += 1
            
        r_greedy, s_next_greedy, EHG, ee_greedy, dr_greedy = env.step_greedy(s_greedy / state_am, sd_greedy, j + i)
        s_greedy = s_next_greedy * state_am
        ep_eh_greedy += EHG
        ep_ee_greedy += ee_greedy
        ep_dr_greedy += dr_greedy

        sd_random, _ = choose_SD(s_random)
        
        # Count SD selections for random
        if sd_random == 0:
            sd0_count_random += 1
        elif sd_random == 1:
            sd1_count_random += 1
            
        r_random, s_next_random, EHR, ee_random, dr_random = env.step_random(s_random / state_am, sd_random, j + i)
        s_random = s_next_random * state_am
        ep_eh_random += EHR
        ep_ee_random += ee_random
        ep_dr_random += dr_random

    # Store the counts for this episode
    sd0_selections_greedy.append(sd0_count_greedy)
    sd1_selections_greedy.append(sd1_count_greedy)
    sd0_selections_random.append(sd0_count_random)
    sd1_selections_random.append(sd1_count_random)
    
    eh_rewardall_greedy.append(ep_eh_greedy / MAX_EP_STEPS)
    eh_rewardall_random.append(ep_eh_random / MAX_EP_STEPS)
    ee_rewardall_greedy.append(ep_ee_greedy / MAX_EP_STEPS)
    ee_rewardall_random.append(ep_ee_random / MAX_EP_STEPS)
    dr_rewardall_greedy.append(ep_dr_greedy / MAX_EP_STEPS)
    dr_rewardall_random.append(ep_dr_random / MAX_EP_STEPS)

    print(f"Episode {i}: Greedy EE={ep_ee_greedy/MAX_EP_STEPS:.4f}, Random EE={ep_ee_random/MAX_EP_STEPS:.4f}")


print("random")
print(f"SUMRATE-> {dr_rewardall_random}")
print("----------------------------------")
print(f"EE-> {ee_rewardall_random}")
print("----------------------------------")
print(f"EH-> {eh_rewardall_random}")
print("----------------------------------")
print("greedy")
print(f"SUMRATE-> {dr_rewardall_greedy}")
print("----------------------------------")
print(f"EE-> {ee_rewardall_greedy}")
print("----------------------------------")
print(f"EH-> {eh_rewardall_greedy}")
print("----------------------------------")

# Print SD selection frequencies
print("SD Selection Frequencies:")
print(f"Greedy - SD 0 selections per episode: {sd0_selections_greedy}")
print(f"Greedy - SD 1 selections per episode: {sd1_selections_greedy}")
print(f"Random - SD 0 selections per episode: {sd0_selections_random}")
print(f"Random - SD 1 selections per episode: {sd1_selections_random}")
print("----------------------------------")
# === PLOTTING ===
# Sum Rate Plot
fig, ax = plt.subplots()
ax.plot(dr_rewardall_greedy, "d-", label='Greedy', linewidth=0.75, color='orange')
ax.plot(dr_rewardall_random, "d-", label='Random', linewidth=0.75, color='black')
ax.set_xlabel("Episodes")
ax.set_ylabel("Average Sum Rate (b/s)")
ax.margins(x=0)
ax.set_xlim(0, MAX_EPISODES)
ax.grid(which="both", axis='y', linestyle=':', color='lightgray', linewidth=0.5)
ax.minorticks_on()
ax.tick_params(which="minor", bottom=False, left=False)
ax.legend(facecolor='none').set_draggable(True)
fig.savefig(f"random_greedy_avg_sumrate_{diversity_mode}.png", bbox_inches="tight")
plt.show()

# Energy Efficiency Plot
fig, ax = plt.subplots()
ax.plot(ee_rewardall_greedy, "v-", label='Greedy', linewidth=0.75, color='orange')
ax.plot(ee_rewardall_random, "v-", label='Random', linewidth=0.75, color='black')
ax.set_xlabel("Episodes")
ax.set_ylabel("Average Energy Efficiency")
ax.margins(x=0)
ax.set_xlim(0, MAX_EPISODES)
ax.grid(which="both", axis='y', linestyle=':', color='lightgray', linewidth=0.5)
ax.minorticks_on()
ax.tick_params(which="minor", bottom=False, left=False)
ax.legend(facecolor='none').set_draggable(True)
fig.savefig(f"random_greedy_avg_ee_{diversity_mode}.png", bbox_inches="tight")
plt.show()
