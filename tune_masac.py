import optuna
import numpy as np
import torch
from masac import MASAC
from env_egc import Env_cellular as env_egc

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
s_dim = (3 * num_SDs) + 1
a_dim = 2
state_am = 1000

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
    scores = []
    for sd in range(num_SDs):
        norm = state[0][sd * 3:(sd + 1) * 3]
        scores.append(np.sum(norm / (np.max(norm) + 1e-8)))
    return np.random.randint(num_SDs) if np.random.rand() < 0.1 else np.argmax(scores)

def objective(trial):
    # Suggest hyperparameters
    lr_a = trial.suggest_float('lr_a', 1e-4, 1e-2, log=True)
    lr_c = trial.suggest_float('lr_c', 1e-4, 1e-2, log=True)
    tau = trial.suggest_float('tau', 0.001, 0.1, log=True)
    gamma = trial.suggest_float('gamma', 0.8, 0.99, step=0.01)
    batch_size = trial.suggest_int('batch_size', 32, 256, step=32)
    memory_capacity = trial.suggest_int('memory_capacity', 5000, 20000, step=1000)
    
    # Set random seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Create environment
    env = env_egc(MAX_EP_STEPS, s_dim, location_PDs, location_SDs, Emax, num_PDs, L, T, eta, Pn, Pmax,
                  w_d, w_egc, w_csk, fading_PD_SD, fading_PD_BS, fading_SD_BS, num_SDs)
    
    # Create MASAC agent with suggested hyperparameters
    # Note: MASAC uses global state dimension for critics and local state dimension for actors
    masac_agent = MASAC(lr_a, lr_c, s_dim, tau, gamma, a_dim, memory_capacity, batch_size, num_SDs)
    
    total_sum_rate = 0
    
    # Training loop
    for i in range(MAX_EPISODES):
        battery = env.reset()
        s = []
        for sd in range(num_SDs):
            s.extend([env.hn_PD_SD[i % num_PDs, sd], env.hn_SD_BS[sd], battery[sd]])
        s.append(env.hn_PD_BS[i % num_PDs])
        s = np.reshape(s, (1, s_dim)) * state_am
        s_masac = s.copy()
        episode_sum_rate = 0

        for j in range(MAX_EP_STEPS):
            sd = choose_SD(s_masac)
            s_local = np.append(s_masac[0][sd * 3:(sd + 1) * 3], s_masac[0][-1])
            a = masac_agent.choose_action(s_local, sd)
            a = np.clip(a[0], 0, 1)
            action_vec = np.zeros(a_dim, dtype=np.float32)
            action_vec[sd] = a

            r, s_next, _, eep, srp = env.step(sd, a, s_masac / state_am, j+i)
            s_next *= state_am
            masac_agent.remember(s_masac[0], action_vec, r, s_next[0])
            masac_agent.learn()
            s_masac = s_next
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
    with open('masac_best_params.txt', 'w') as f:
        f.write(f"Best sum rate: {trial.value}\n")
        f.write("Best parameters:\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main() 