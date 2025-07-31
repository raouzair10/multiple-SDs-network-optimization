import optuna
import numpy as np
import torch
from mappo import MAPPO
from env_egc import Env_cellular as env_egc
from buffer import RolloutBuffer

# === Fixed Hyperparameters ===
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
MAX_EPISODES = 200
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 10000
state_am = 1000

s_dim = (3 * num_SDs) + 1
a_dim = num_SDs

# === Topology and Channel ===
location_PDs = np.array([[0, 1], [0, 1000]])
location_SDs = np.array([[1, 1], [1, 1000]])
fading_SD_BS = np.ones(num_SDs)
fading_PD_BS = np.array([0.79643, 0.87451])
hnx = np.array([[[0.90877, 0.80627], [0.82154, 0.96098]],
                [[0.6277, 0.63398], [0.60789, 0.92472]]])
fading_PD_SD = np.zeros((num_PDs, num_SDs))

# Setup EGC diversity
for i in range(num_PDs):
    for j in range(num_SDs):
        fading_PD_SD[i][j] = np.sum(hnx[i][j]) ** 2

def choose_SD(state):
    sd_qualities = []
    for sd in range(num_SDs):
        sd_state = state[0][sd * 3:(sd + 1) * 3]
        norm = (sd_state - np.min(sd_state)) / (np.max(sd_state) - np.min(sd_state) + 1e-8)
        quality = np.sum(norm)
        sd_qualities.append(quality)
    return np.random.randint(num_SDs) if np.random.rand() < 0.1 else np.argmax(sd_qualities)

def objective(trial):
    # Suggest hyperparameters
    gamma = trial.suggest_float('gamma', 0.8, 0.99, step=0.01)
    eps_clip = trial.suggest_float('eps_clip', 0.1, 0.5, step=0.05)
    K_epochs = trial.suggest_int('K_epochs', 3, 10)
    lr_actor = trial.suggest_float('lr_actor', 1e-4, 1e-2, log=True)
    lr_critic = trial.suggest_float('lr_critic', 1e-4, 1e-2, log=True)
    action_std_init = trial.suggest_float('action_std_init', 0.1, 0.5, step=0.05)
    buffer_size = trial.suggest_int('buffer_size', 1000, 10000, step=1000)
    
    # Set random seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Create environment
    env = env_egc(MAX_EP_STEPS, s_dim, location_PDs, location_SDs, Emax, num_PDs, L, T, eta, Pn, Pmax,
                  w_d, w_egc, w_csk, fading_PD_SD, fading_PD_BS, fading_SD_BS, num_SDs)
    
    # Create MAPPO agent with suggested hyperparameters
    mappo_agent = MAPPO(num_SDs, 4, s_dim, 1, action_std_init, gamma, eps_clip, K_epochs, buffer_size, lr_actor, lr_critic)
    
    total_sum_rate = 0
    
    # Training loop
    for ep in range(MAX_EPISODES):
        battery = env.reset()
        s = []
        for sd in range(num_SDs):
            s.extend([env.hn_PD_SD[ep % num_PDs, sd], env.hn_SD_BS[sd], battery[sd]])
        s.append(env.hn_PD_BS[ep % num_PDs])
        s = np.reshape(s, (1, s_dim)) * state_am
        s_mappo = s.copy()
        episode_sum_rate = 0

        for step in range(MAX_EP_STEPS):
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
            episode_sum_rate += srp

        total_sum_rate += episode_sum_rate / MAX_EP_STEPS
    
    # Return average sum rate (maximize this)
    return total_sum_rate / MAX_EPISODES

def main():
    # Create study
    study = optuna.create_study(direction='maximize')
    
    # Run optimization with trial callback
    def print_trial_result(study, trial):
        print(f"Trial {trial.number}: Sum Rate = {trial.value:.6f}")
        print(f"  Parameters: {trial.params}")
        print()
    
    study.optimize(objective, n_trials=30, callbacks=[print_trial_result])
    
    # Print best results
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save results
    with open('mappo_best_params.txt', 'w') as f:
        f.write(f"Best sum rate: {trial.value}\n")
        f.write("Best parameters:\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main() 