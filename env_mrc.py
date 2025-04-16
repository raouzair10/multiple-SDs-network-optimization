import numpy as np
from scipy.special import lambertw
import math
dtype = np.float32

class Env_cellular():
    def __init__(self, MAX_EP_STEPS, s_dim, location_PDs, location_SDs, Emax, num_PDs, T, eta, Pn, Pmax, w_d, w_mrc, w_csk, fading_PD_SD, fading_PD_BS, fading_SD_BS, num_SDs):
        self.emax = Emax        # Maximum battery capacity for SD
        self.num_PDs = num_PDs  # Number of Primary Devices (PDs)
        self.T = T              # Time duration for each time slot
        self.eta = eta          # Energy harvesting efficiency (between 0 and 1)
        self.Pn = Pn            # transmit power of PDs
        self.Pmax = Pmax        # Maximum transmit power of the SD
        self.w_csk = w_csk      # circuit and signal processing power for the SD
        self.w_d = w_d
        self.w_mrc = w_mrc
        self.s_dim = s_dim      # dimension of states
        self.MAX_EP_STEPS = MAX_EP_STEPS  # Max steps per episode
        self.num_SDs = num_SDs  # Number of Secondary Devices

        BW = 10 ** 6    # Bandwidth of 1 MHz
        sigma2_dbm = -170 + 10 * np.log10(BW)           # Thermal noise power in dBm, considering the bandwidth
        self.noise = 10 ** ((sigma2_dbm - 30) / 10)     # Noise power in linear scale (Watts)

        distance_PD_SD = np.sqrt(np.sum((location_PDs[:, np.newaxis, :] - location_SDs[np.newaxis, :, :]) ** 2, axis=2))  # Distance from each PD to each SD
        # [[893.65373607 804.39293881]
        #  [892.06782253 803.05666052]]
        # [[PD1-SD1 PD2-SD1]
        #  [PD1-SD2 PD2-SD2]]

        distance_PD_BS = np.sqrt(np.sum((location_PDs) ** 2, axis=1))   # Distance from each PD user to the base station
        # [900.84904396 811.38153787]

        distance_SD_BS = np.sqrt(np.sum(location_SDs ** 2, axis=1))
        # [7.21110255 8.94427191]

        PL_alpha = 3    # Path loss exponent, typically between 2 and 4
        PL_PD_SD = 1 / (distance_PD_SD ** PL_alpha) / (10 ** 3.17)  # Path loss for PD-SD links
        # [[9.47310114e-13 1.29895843e-12]
        #  [9.52371470e-13 1.30545358e-12]]
        # [[PD1-SD1 PD2-SD1]
        #  [PD1-SD2 PD2-SD2]]

        PL_PD_BS = 1 / (distance_PD_BS ** PL_alpha) / (10 ** 3.17)  # Path loss for PD-BS links
        # [9.24791723e-13 1.26568209e-12]

        PL_SD_BS = 1 / (distance_SD_BS ** PL_alpha) / (10 ** 3.17) # Path loss for SD-BS links
        # [1.80299692e-06 9.44854682e-07]

        self.hn_PD_SD = np.multiply(PL_PD_SD, fading_PD_SD) # Effective channel gains for PD-SD links
        # [[7.10046823e-13 8.65486621e-13]
        #  [1.24827307e-12 8.27631463e-13]]
        # [[PD1-SD1 PD1-SD2]
        #  [PD2-SD1 PD2-SD2]]

        self.hn_PD_BS = fading_PD_BS * PL_PD_BS  # Effective channel gains for PD-BS links
        # [7.36531872e-13 1.10685165e-12]

        self.hn_SD_BS = fading_SD_BS * PL_SD_BS # Effective channel gains for SD-BS links
        # [1.80299692e-06 9.44854682e-07]

        self.h0 = self.hn_SD_BS / self.noise    # Normalized channel gains for SD-BS links
        # [1.80299692e+08 9.44854682e+07]

    def step(self, actions, state, j): # state: [[hn-SD1-PDi hn-SD1-BS battery-SD1 hn-SD2-PDi hn-SD2-BS battery-SD2 hn-PDi-BS]]
        active_sd = int(actions[0])
        action = actions[1]
        hn = []             # Normalized PD-SD channel gains
        h0 = []             # Normalized SD-BS channel gains
        En = []             # Battery levels
        hn0 = state[0, -1]  # PD-BS channel gain
        En_bar = [0]*self.num_SDs

        for i in range(0, self.num_SDs*3, 3):
            hn.append(state[0, i] / self.noise)
            h0.append(state[0, i+1] / self.noise)
            En.append(state[0, i+2])

        for sd_index in range(self.num_SDs):
            if sd_index == active_sd:     # Net change in battery energy (harvested - consumed)
                En_bar[sd_index] = action * min(self.emax - En[sd_index], (self.T * self.eta * self.Pn * hn0) - self.w_mrc - self.w_d) - (1 - action) * min(En[sd_index], self.T * self.Pmax + self.T * self.w_csk)
            else:                   # Net change in battery energy (only harvested)
                En_bar[sd_index] = min(self.emax - En[sd_index], (self.T * self.eta * self.Pn * hn0) - self.w_mrc - self.w_d)

        # Intermediate Variables for Reward Calculation
        mu1 = self.eta * self.Pn * hn0 * h0[active_sd] / (1 + self.Pn * hn[active_sd]) # Represents the impact of energy harvesting on potential transmission power
        mu2 = (En_bar[active_sd] + self.w_mrc + self.w_d) * h0[active_sd] / (self.T * (1 + self.Pn * hn[active_sd]))          # Adjusts for energy change per unit time
        mu3 = self.w_csk * h0[active_sd] / (1 + self.Pn * hn[active_sd])               # Impact of circuit power consumption
        
        # Calculate Optimal Time Allocation
        wx0 = np.real(lambertw(math.exp(-1) * (1 - mu1 - mu3), k=0))
        alphaxx = (mu1 - mu2) / (math.exp(wx0 + 1) - 1 + mu1 + mu3)
        alpha01 = ((self.T * self.eta * self.Pn * hn0) - self.w_mrc - self.w_d - En_bar[active_sd]) / (self.T * self.eta * self.Pn * hn0 + self.T * self.w_csk + self.T * self.Pmax)
        alpha02 = ((self.T * self.eta * self.Pn * hn0) - self.w_mrc - self.w_d - En_bar[active_sd] - En[active_sd]) / ((self.T * self.eta * self.Pn * hn0) - self.T * self.w_csk)
        alphax2 = max(alpha01, alpha02)
        alphan = min(1, max(alphaxx, alphax2))  # Optimal fraction of time allocated for data transmission
        
        # Compute Reward
        reward = 0
        if En_bar[active_sd] == self.T * self.eta * self.Pn * hn0:
            P0n=0
            reward = 0  # If no energy is available or allocated, reward is zero
        elif alphan == 0: #<= 0.00000001:
            P0n=0
            reward = 0
        else:
            # Compute transmission power (P0n) and reward based on the Shannon capacity formula
            P0n = (1 - alphan) * self.eta * self.Pn * hn0 - self.w_mrc - self.w_d / alphan - En_bar[active_sd] / alphan / self.T - self.w_csk / self.T
            if P0n < 0:
                P0n = 0
            reward = alphan * np.log(1 + P0n * h0[active_sd] / (1 + self.Pn * hn[active_sd]))

        # Handle NaN Rewards
        if math.isnan(reward):
            reward = 0

        # Energy Efficiency Calculation
        data_rate = alphan * np.log(1 + P0n * h0[active_sd] / (1 + self.Pn * hn[active_sd]))  # Shannon's capacity formula
        energy_consumed = (P0n + self.w_csk) * self.T  # Energy consumed
        energy_efficiency = data_rate / energy_consumed  # Data rate per unit energy

        batter_new = []
        for s in range(self.num_SDs):
            batter_new.append(min(self.emax, En[s] + En_bar[s]))    # Updated battery level after accounting for energy harvested and consumed
                
        # Next state, including updated channel gains and battery level
        state_next = []
        for sd in range(self.num_SDs):
            state_next.append(self.hn_PD_SD[(j+1) % self.num_PDs, sd])
            state_next.append(self.hn_SD_BS[sd])
            state_next.append(batter_new[sd])
        state_next.append(self.hn_PD_BS[(j+1) % self.num_PDs])
        state_next = np.reshape(state_next, (1, self.s_dim))

        # Represents the actual energy harvested during the time allocated for energy harvesting
        EHD = (1 - alphan) * self.eta * self.T * self.Pn * hn0

        return reward, state_next, EHD, energy_efficiency

    def step_greedy(self, state, active_sd, j):
        hn = []             # Normalized PD-SD channel gains
        h0 = []             # Normalized SD-BS channel gains
        En = []             # Battery levels
        hn0 = state[0, -1]  # PD-BS channel gain
        En_bar = [0]*self.num_SDs

        for i in range(0, self.num_SDs*3, 3):
            hn.append(state[0, i] / self.noise)
            h0.append(state[0, i+1] / self.noise)
            En.append(state[0, i+2])

        alphan = min(1, En[active_sd]/self.T/self.Pmax) # Allocates maximum possible time for data transmission without exceeding battery capacity

        for sd_index in range(self.num_SDs):
            if sd_index == active_sd:     # Net change in battery energy (harvested - consumed)
                En_bar[sd_index] = alphan * min(self.emax - En[sd_index], (self.T * self.eta * self.Pn * hn0) - self.w_mrc - self.w_d) - (1 - alphan) * min(En[sd_index], self.T * self.Pmax + self.T * self.w_csk)
            else:                   # Net change in battery energy (only harvested)
                En_bar[sd_index] = min(self.emax - En[sd_index], (self.T * self.eta * self.Pn * hn0) - self.w_mrc - self.w_d)

        # Compute Reward
        reward = 0
        # Compute Reward
        if alphan==0:
            reward = 0
        else:
            reward = alphan * np.log(1 + self.Pmax*h0[active_sd] / (1 + self.Pn * hn[active_sd]))

        # Handle NaN Rewards
        if math.isnan(reward):
            reward = 0

        # Energy Efficiency Calculation
        data_rate = alphan * np.log(1 + self.Pmax * h0[active_sd] / (1 + self.Pn * hn[active_sd]))  # Shannon's capacity formula
        energy_consumed = (self.Pmax + self.w_csk) * self.T  # Energy consumed
        energy_efficiency = data_rate / energy_consumed  # Data rate per unit energy

        batter_new = []
        for s in range(self.num_SDs):
            batter_new.append(min(self.emax, En[s] + En_bar[s]))    # Updated battery level after accounting for energy harvested and consumed
                
        # Next state, including updated channel gains and battery level
        state_next = []
        for sd in range(self.num_SDs):
            state_next.append(self.hn_PD_SD[(j+1) % self.num_PDs, sd])
            state_next.append(self.hn_SD_BS[sd])
            state_next.append(batter_new[sd])
        state_next.append(self.hn_PD_BS[(j+1) % self.num_PDs])
        state_next = np.reshape(state_next, (1, self.s_dim))

        # Represents the actual energy harvested during the time allocated for energy harvesting
        EHG = (1 - alphan) * self.eta * self.T * self.Pn * hn0

        return reward, state_next, EHG, energy_efficiency

    def step_random(self, state, active_sd, j):
        hn = []             # Normalized PD-SD channel gains
        h0 = []             # Normalized SD-BS channel gains
        En = []             # Battery levels
        hn0 = state[0, -1]  # PD-BS channel gain
        En_bar = [0]*self.num_SDs

        for i in range(0, self.num_SDs*3, 3):
            hn.append(state[0, i] / self.noise)
            h0.append(state[0, i+1] / self.noise)
            En.append(state[0, i+2])

        alphan = np.random.uniform(0, min(1, En[active_sd]/self.T/self.Pmax)) # Randomly Select Transmission Time (alphan)

        for sd_index in range(self.num_SDs):
            if sd_index == active_sd:     # Net change in battery energy (harvested - consumed)
                En_bar[sd_index] = alphan * min(self.emax - En[sd_index], (self.T * self.eta * self.Pn * hn0) - self.w_mrc - self.w_d) - (1 - alphan) * min(En[sd_index], self.T * self.Pmax + self.T * self.w_csk)
            else:                   # Net change in battery energy (only harvested)
                En_bar[sd_index] = min(self.emax - En[sd_index], (self.T * self.eta * self.Pn * hn0) - self.w_mrc - self.w_d)

        # Compute Reward
        reward = 0
        # Compute Reward
        if alphan==0:
            reward = 0
        else:
            reward = alphan * np.log(1 + self.Pmax*h0[active_sd] / (1 + self.Pn * hn[active_sd]))

        # Handle NaN Rewards
        if math.isnan(reward):
            reward = 0

        # Energy Efficiency Calculation
        data_rate = alphan * np.log(1 + self.Pmax * h0[active_sd] / (1 + self.Pn * hn[active_sd]))  # Shannon's capacity formula
        energy_consumed = (self.Pmax + self.w_csk) * self.T  # Energy consumed
        energy_efficiency = data_rate / energy_consumed  # Data rate per unit energy

        batter_new = []
        for s in range(self.num_SDs):
            batter_new.append(min(self.emax, En[s] + En_bar[s]))    # Updated battery level after accounting for energy harvested and consumed
                
        # Next state, including updated channel gains and battery level
        state_next = []
        for sd in range(self.num_SDs):
            state_next.append(self.hn_PD_SD[(j+1) % self.num_PDs, sd])
            state_next.append(self.hn_SD_BS[sd])
            state_next.append(batter_new[sd])
        state_next.append(self.hn_PD_BS[(j+1) % self.num_PDs])
        state_next = np.reshape(state_next, (1, self.s_dim))

        # Represents the actual energy harvested during the time allocated for energy harvesting
        EHG = (1 - alphan) * self.eta * self.T * self.Pn * hn0

        return reward, state_next, EHG, energy_efficiency

    def reset(self):
        batter_ini = [self.emax] * self.num_SDs
        return batter_ini
print("MRC Env Successfully Compiled")