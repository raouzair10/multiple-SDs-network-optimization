import numpy as np
from scipy.special import lambertw
import math
dtype = np.float32

class Env_cellular():
    def __init__(self, MAX_EP_STEPS, s_dim, location_PDs, location_SDs, Emax, num_PDs, T, eta, Pn, Pmax, w_csk, fading_n, fading_0):
        self.emax = Emax        # Maximum battery capacity for SD
        self.num_PDs = num_PDs  # Number of Primary Devices (PDs)
        self.T = T              # Time duration for each time slot
        self.eta = eta          # Energy harvesting efficiency (between 0 and 1)
        self.Pn = Pn            # transmit power of PDs
        self.Pmax = Pmax        # Maximum transmit power of the SD
        self.w_csk = w_csk      # circuit and signal processing power for the SD
        self.s_dim = s_dim      # dimsion of states
        self.MAX_EP_STEPS = MAX_EP_STEPS  # Max steps per episode
        
        BW = 10 ** 6    # Bandwidth of 1 MHz
        sigma2_dbm = -170 + 10 * np.log10(BW)           # Thermal noise power in dBm, considering the bandwidth
        self.noise = 10 ** ((sigma2_dbm - 30) / 10)     # Noise power in linear scale (Watts)
        # self.noise is 1e-14

        distance_SD_PD = np.sqrt(np.sum((location_PDs - location_SDs) ** 2, axis=1))   # Distance from each PD user to the SD user
        # distance_SD_PD   is    [1.41421356 2.23606798]
        # distance_SD_PD is like [PD1-SD1 PD2-SD1]

        distance_BS_PD = np.sqrt(np.sum((location_PDs) ** 2, axis=1))                  # Distance from each PD user to the base station
        # distance_BS_PD   is    [8.60232527 7.81024968]
        # distance_BS_PD is like [PD1-BS PD2-BS]

        distance = np.matrix.transpose(np.vstack((distance_SD_PD, distance_BS_PD)))    # Combined distances in a matrix
        distance = np.maximum(distance, np.ones((self.num_PDs, 2)))                    # Ensures minimum distance of 1 meter to avoid singularities
        # distance    is   [[1.41421356 8.60232527]
        #                  [2.23606798 7.81024968]]
        # distance is like [[PD1-SD1 PD1-BS]
        #                  [PD2-SD1 PD2-BS]]

        PL_alpha = 3                                    # Path loss exponent, typically between 2 and 4
        PL = 1 / (distance ** PL_alpha) / (10 ** 3.17)  # Path loss for each link, considering the distance and path loss exponent
        # PL    is   [[2.39031428e-04 1.06206824e-06]
        #            [6.04706997e-05 1.41907467e-06]]
        # PL is like [[PD1-SD1 PD1-BS]
        #            [PD2-SD1 PD2-BS]]
        
        self.hn = np.multiply(PL, fading_n)     # Effective channel gains, incorporating path loss and fading (element-wise multiply)
        # self.hn    is   [[1.84211961e-04 8.45863005e-07]
        #                  [4.53252082e-05 1.24099499e-06]]
        # self.hn is like [[PD1-SD1 PD1-BS]
        #                  [PD2-SD1 PD2-BS]]

        distance_SD_BS = np.maximum(np.sqrt(np.sum(location_SDs ** 2, axis=1)), 1)    # Distance from SD to BS
        # distance_SD_BS is [7.21110255]

        PL0 = 1 / (distance_SD_BS ** PL_alpha) / (10 ** 3.17)
        # PL0 is [1.80299692e-06]

        self.h0 = (fading_0 * PL0) / self.noise   # Normlized channel gain between SD and the BS
        # self.h0 is [1.80299692e+08]

        print(self.hn)

    # Calculates the next state, reward, energy harvested, and whether the episode is done
    # Parameters:
        # action: The agent's action, a value between 0 and 1 indicating the proportion of time spent on energy harvesting.
        # state: Current state vector.
        # j: Current time step index
    def step(self, action, state, j):
        hn = state[0, 0] / self.noise   # Normalized PD-SD channel gain
        hn0 = state[0, 1]               # PD-BS channel gain
        En = state[0, -1]               # Current battery level

        # Net change in battery energy (harvested - consumed)
        En_bar = action * min(self.emax - En, self.T * self.eta * self.Pn * hn0) - (1 - action) * min(En, self.T * self.Pmax + self.w_csk)
        
        # Intermediate Variables for Reward Calculation
        mu1 = self.eta * self.Pn * hn0 * self.h0 / (1 + self.Pn * hn) # Represents the impact of energy harvesting on potential transmission power
        mu2 = En_bar * self.h0 / self.T / (1 + self.Pn * hn)          # Adjusts for energy change per unit time
        mu3 = self.w_csk * self.h0 / (1 + self.Pn * hn)               # Impact of circuit power consumption
        
        # Calculate Optimal Time Allocation
        wx0 = np.real(lambertw(math.exp(-1) * (mu1 - 1), k=0))
        alphaxx = (mu1 - mu2) / (math.exp(wx0 + 1) - 1 + mu1 + mu3)
        alpha01 = 1 - (En + En_bar) / (self.T * self.eta * self.Pn * hn0)
        alpha02 = (self.T * self.eta * self.Pn * hn0 - En_bar) \
                  / (self.T * self.eta * self.Pn * hn0 + self.T * self.Pmax + self.T * self.w_csk)
        alphax2 = max(alpha01, alpha02)
        alphan = min(1, max(alphaxx, alphax2))  # Optimal fraction of time allocated for data transmission
        
        # Compute Reward
        reward = [0]
        if En_bar == self.T * self.eta * self.Pn * hn0:
            P0n=0
            reward = [0]  # If no energy is available or allocated, reward is zero
        elif alphan == 0: #<= 0.00000001:
            P0n=0
            reward = [0]
        else:
            # Compute transmission power (P0n) and reward based on the Shannon capacity formula
            P0n = (1 - alphan) * self.eta * self.Pn * hn0 / alphan - En_bar / alphan / self.T - self.w_csk
            if P0n < 0:
                P0n = 0
            reward = alphan * np.log(1 + P0n * self.h0 / (1 + self.Pn * hn))

        # Handle NaN Rewards
        if math.isnan(reward[0]):
            reward = [0]

        # Initialize data_rate and energy_consumed with default values
        data_rate = 0
        energy_consumed = 0

        # Energy Efficiency Calculation
        data_rate = alphan * np.log(1 + P0n * self.h0 / (1 + self.Pn * hn))  # Shannon's capacity formula
        energy_consumed = (P0n + self.w_csk) * self.T  # Energy consumed
        energy_efficiency = data_rate / energy_consumed  # Data rate per unit energy

        batter_new = min(self.emax, En + En_bar) # Updated battery level after accounting for energy harvested and consumed
        
        # Next state, including updated channel gains and battery level
        state_next = self.hn[(j+1) % self.num_PDs, :].tolist()
        state_next.append(float(batter_new))
        state_next = np.reshape(state_next, (1, self.s_dim))

        # Represents the actual energy harvested during the time allocated for energy harvesting
        EHD = (1 - alphan) * self.eta * self.T * self.Pn * hn0

        done=False  # Episode continues     TODO: not required
        return reward, state_next, EHD, done, energy_efficiency,

    # Simulates a greedy policy where the GF user transmits as much as possible given the battery constraints
    def step_greedy(self,   state, j):
        hn = state[0,0]/self.noise
        hn0 = state[0,1]
        En = state[0,-1]
        alphan = min(1, En/self.T/self.Pmax) # Allocates maximum possible time for data transmission without exceeding battery capacity
        
        # Compute Reward
        if alphan==0:
            reward = 0
        else:
            reward = alphan*np.log(1 + self.Pmax*self.h0/(1 + self.Pn*hn))

        if math.isnan(reward):
            print(f"alpha is {alphan}  ")
            print(f"{En}")
        
        # Energy Efficiency Calculation
        if alphan > 0 and self.Pmax > 0:
            data_rate = alphan * np.log(1 + self.Pmax * self.h0 / (1 + self.Pn * hn))  # Shannon's capacity formula
            energy_consumed = (self.Pmax * alphan + self.w_csk) * self.T  # Energy consumed
            if energy_consumed > 0:
                energy_efficiency = data_rate / energy_consumed  # Data rate per unit energy
            else:
                energy_efficiency = 0
        else:
            energy_efficiency = 0
        
        # Update Battery Level
        batter_new = min(self.emax, En - alphan*self.T*self.Pmax + (1-alphan) * self.T*self.eta*self.Pn*hn0)
        
        # Calculate Energy Harvested
        EHG = (1-alphan)*self.T*self.eta*self.Pn*hn0
        
        # Prepare Next State and Return Values
        state_next = self.hn[(j+1) % self.num_PDs, :].tolist()
        state_next.append(batter_new)
        state_next = np.reshape(state_next, (1, self.s_dim))
        done = False
        return reward, state_next, EHG, done, energy_efficiency,

    # Simulates a random policy where alphan is chosen randomly within feasible limits
    def step_random(self,   state, j):
        hn = state[0,0]/self.noise
        hn0 = state[0,1]
        En = state[0,-1]
        alphan = np.random.uniform(0, min(1, En/self.T/self.Pmax)) # Randomly Select Transmission Time (alphan)
        
        # Compute Reward
        if alphan==0:
            reward = 0
        else:
            reward = alphan*np.log(1 + self.Pmax*self.h0/(1 +self.Pn*hn))

        if math.isnan(reward):
            print(f"alpha is {alphan}  ")
            print(f"{En}")
        
        # Energy Efficiency Calculation
        if alphan > 0 and self.Pmax > 0:
            data_rate = alphan * np.log(1 + self.Pmax * self.h0 / (1 + self.Pn * hn))  # Shannon's capacity formula
            energy_consumed = (self.Pmax * alphan + self.w_csk) * self.T  # Energy consumed
            if energy_consumed > 0:
                energy_efficiency = data_rate / energy_consumed  # Data rate per unit energy
            else:
                energy_efficiency = 0
        else:
            energy_efficiency = 0
        
        # Update Battery Level
        batter_new = min(self.emax, En -alphan*self.T*self.Pmax +(1-alphan)*self.T*self.eta*self.Pn*hn0)
        
        # Calculate Energy Harvested
        EHR = (1-alphan)*self.T*self.eta*self.Pn*hn0
        
        # Prepare Next State and Return Values
        state_next = self.hn[(j+1) % self.num_PDs, :].tolist()
        state_next.append(batter_new)
        state_next = np.reshape(state_next, (1, self.s_dim))
        done=False
        return reward, state_next, EHR, done, energy_efficiency,

    # Resets the environment to its initial state with the GF user's battery fully charged
    def reset(self):
        batter_ini = self.emax
        return batter_ini
print("Simple Env Successfully Compiled")