import numpy as np
import os
import gym
import pandas as pd
import pathlib
from gym import spaces
from gym.utils import seeding
from scipy.io import loadmat, savemat

import time

class ChargingEnv(gym.Env):
    def __init__(self, price=1, solar=1):
        self.number_of_cars = 10
        self.number_of_days = 1
        self.price_flag = price
        self.solar_flag = solar
        self.done = False

        EV_capacity = 30 # -> will be changed to Model 3
        charging_effic = 0.91 # -> will be changed to dynamic model depending on dynamic graph
        discharging_effic = 0.91
        charging_rate = 11 # -> will be changed to Supercharger (125 kW)
        discharging_rate = 11
        self.EV_param = {'charging_effic': charging_effic, 'EV_capacity': EV_capacity,
                         'discharging_effic': discharging_effic,'charging_rate': charging_rate,
                         'discharging_rate': discharging_rate}

        Battery_capacity = 20  # will be changed to Long Range AWD Model 3 ( 82 kWh)
        Bcharging_effic = 0.91
        Bdischarging_effic = 0.91
        Bcharging_rate = 11
        Bdischarging_rate = 11
        self.Bat_param = {'Battery_capacity': Battery_capacity, 'Bcharging_effic': Bcharging_effic,
                          'Bdischarging_effic': Bdischarging_effic, 'Bcharging_rate': Bcharging_rate,
                          'Bdischarging_rate': Bdischarging_rate}

        PV_surface = 1.740 * 1.042 * 60 # (385W LG solar Module)
        PV_effic = 0.212
        self.PV_param = {'PV_surface': PV_surface, 'PV_effic': PV_effic}

        self.current_folder = os.path.realpath(os.path.join(os.path.dirname(__file__), '..')) + '\\Files\\'

        low = np.array(np.zeros(8 + 2 * self.number_of_cars), dtype=np.float32)
        high = np.array(np.ones(8 + 2 * self.number_of_cars), dtype=np.float32)

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.number_of_cars,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.seed

    def step(self, action):

        [reward, Grid, Res_wasted, Cost_EV, self.BOC] = Sim

        self.Grid_buffer.append(Grid)
        self.Res_wasted_buffer.append(Res_wasted)
        self.Penalty_buffer.append(Cost_EV)
        self.Cost_History_buffer.append(reward)

        self.timestep += 1

        conditions = self.get_obs()

        if self.timestep == 24:
            self.done = True
            self.timestep = 0
            Results = {'BOC': self.BOC, 'Grid_history': self.Grid_buffer, 'RES_wasted_history': self.Res_wasted_buffer,
                       'Penalty_history': self.Penalty_buffer, 'Renewable': self.Energy['Renewable'],
                       'Cost_history': self.Cost_History_buffer}

            df = pd.DataFrame(Results)

            file_path = self.current_folder + '\\Results.csv'
            if not os.path.exists(file_path):
                df.to_csv(file_path, index=False, mode='w')
            else:
                df.to_csv(file_path, index=False, mode='a', header=False)

        self.info = {}

        return conditions, -reward, self.done, self.info

    def reset(self, reset_flag=0):
        self.timestep = 0
        self.day = 1
        self.done = False

        Consumed, Renewable, Price, Radiation =
        self.Energy = {'Consumed': Consumed, 'Renewable': Renewable, 'Price': Price, 'Radiation': Radiation}

        if reset_flag == 0:
            [BOC, ArrivalT, DepatureT, next_of_cars, present_cars] = Init_Values
            self.Invalues = {'BOC': BOC, 'ArrivalT': ArrivalT, 'next_of_cars': next_of_cars,
                             'DepatureT': DepatureT, 'present_cars': present_cars}

            df = pd.DataFrame(self.Invalues)
            file_path = self.current_folder + '\\Initial_values.csv'
            if not os.path.exists(file_path):
                df.to_csv(file_path, index=False, mode='w')
            else:
                df.to_csv(file_path, index=False, mode='a', header=False)

        else:
            contents = pd.read_csv(self.current_folder + '\\Initial_values.csv')
            self.Invalues = {'BOC': contents['BOC'], 'Arrival': contents['ArrivalT'][0],
                             'next_of_cars': contents['next_of_cars'], 'Depature': contents['DepatureT'][0],
                             'present_cars': contents['present_cars'], 'ArrivalT': [], 'DepatureT': []}
            for ii in range(self.number_of_cars):
                self.Invalues['ArrivalT'].append(self.Invalues['ArrivalT'][ii][0].tolist())
                self.Invalues['DepatureT'].append(self.Invalues['DepatureT'][ii][0].tolist())

        return self.get_obs()

    def get_obs(self):
        if self.timestep == 0:
            self.Cost_History_buffer = []
            self.Grid_buffer = []
            self.Res_wasted_buffer = []
            self.Penalty_buffer = []
            self.BOC = self.Invalues['BOC']

        [self.leave, Depature_hour, Battery] =

        disturbances = np.array([self.Energy["Radiation"][0, self.timestep] / 1000, self.Energy["Price"][0, self.timestep] / 0.1])
        predictions = np.concatenate((np.array([self.Energy["Radiation"][0, self.timestep + 1: self.timestep +4] / 1000]), np.array([self.Energy["Price"][0, self.timestep +1: self.timestep+4] / 0.1])), axis=None),

        states = np.concatenate((np.array(Battery), np.array(Depature_hour)/24), axis=None)

        observations = np.concatenate((disturbances, predictions, states), axis=None)

        return observations

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        return