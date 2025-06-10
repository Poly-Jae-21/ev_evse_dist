import datetime
import json
import pickle
from random import random

import gymnasium as gym
import numpy as np
import yaml
import random
import os
from gym import spaces
from numpy import dtype

from pyvista import Renderer

from VDN.common.loaders import load_transformers


def SquaredTrackingErrorReward(env, *args):
    reward = -(min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) - env.current_power_usage[env.current_step-1]) ** 2

    return reward

def PublicPST(env, *args):
    state = [(env.current_step/env.simulation_length)]
    if env.current_step < env.simulation_length:
        setpoint = env.power_setpoints[env.current_step]
    else:
        setpoint = np.zeros((1))

    state.append(setpoint)
    state.append(env.current_power_usage[env.current_step-1])

    for tr in env.transformer:
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:
                for EV in cs.evs_connected:
                    if EV is not None:
                        state.append([
                            1 if EV.get_soc() == 1 else 0.5,
                            EV.total_energy_exchanged,
                            (env.current_step - EV.time_of_arrival)
                        ])
                    else:
                        state.append(np.zeros(3))

    state = np.array(np.hstack(state))

    np.set_printoptions(suppress=True)

    return state

class ev2gym(gym.Env):
    def __init__(self, config_file=None, generate_rnd_game=True, load_from_replay_path=None, empty_ports_at_end_of_simulation=True, save_replay=False, lightweights_plots=False, eval_mode="Normal", render_mode=None, replay_save_path='./replay/', cost_function=None, seed=None, extra_sim_name=None, verbose=False):
        super(gym.Env, self).__init__()

        if verbose:
            print(f"Initializing env2gym environment")

        assert config_file is not None, "Please provide a config file"
        self.config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)

        self.generate_rnd_game = generate_rnd_game
        self.load_from_replay_path = load_from_replay_path
        self.empty_ports_at_end_of_simulation = empty_ports_at_end_of_simulation # Whether to empty the ports at the end of the simulation or not
        self.save_replay = save_replay
        self.lightweight_plots = lightweights_plots
        self.eval_mode = eval_mode # eval model can be "Normal", "Unstirred" or "Optimal" in order to save the correct statistics in the replay file
        self.verbose = verbose # Whether to print the simulation progress or not
        self.render_mode =render_mode # Wether to render the simulation in real-time or not

        self.simulation_length = self.config['simulation_length']

        self.replay_path = replay_save_path

        cs = self.config['number_of_charging_stations']

        self.reward_function = SquaredTrackingErrorReward
        self.state_function = PublicPST
        self.cost_function = cost_function

        if seed is None:
            self.seed = np.random.randint(0, 1000000)
        else:
            self.seed = seed

        np.random.seed(self.seed)
        random.seed(self.seed)

        self.tr_seed = self.config['tr_seed']
        if self.tr_seed == -1:
            self.tr_seed = self.seed
        self.tr_rng = np.random.default_rng(seed=self.tr_seed)

        if load_from_replay_path is not None:
            with open(load_from_replay_path, 'rb') as file:
                self.replay = pickle.load(file)

            sim_name = self.replay.replay_path.split('replay_')[-1].split('.')[0]
            self.sim_name = sim_name + '_replay'
            self.sim_date = self.replay.sim_date
            self.timescale = self.replay.timescale
            self.cs = self.replay.n_cs
            self.number_of_transformers = self.replay.n_transformers
            self.number_of_ports_per_cs = self.replay_max_n_ports
            self.scenario = self.replay.scenario
            self.heterogenous_specs = self.replay.heterogenous_specs

        else:
            assert cs is not None, "Please provide the number of charging stations"
            self.cs = cs

            self.number_of_ports_per_cs = self.config['number_of_ports_per_cs']
            self.number_of_transformers = self.config['number_of_transformers']
            self.timescale = self.config['timescale']
            self.scenario = self.config['scenario']
            self.simulation_length = self.config['simulation_length']

            if self.config['random_day']:
                if "random_hour" in self.config:
                    self.config['hour'] = random.randint(5,15)

                self.sim_date = datetime.datetime(2025,1,1,self.config['hour'], self.config['minute']) + datetime.timedelta(days=random.randint(0, int(1.5*365)))

                if self.scenario == 'workplace':
                    while self.sim_date.weekday() > 4:
                        self.sim_date += datetime.timedelta(days=1)

                if self.config['simulation_days'] == 'weekdays':
                    self.sim_date += datetime.timedelta(days=1)

            else:
                self.sim_date = datetime.datetime(self.config['year'],
                                                  self.config['month'],
                                                  self.config['day'],
                                                  self.config['hour'],
                                                  self.config['minute'])

            self.replay = None
            self.sim_name = f'sim_' + \
                f'{datetime.datetime.now().strftime("%Y_%m_%d_%f")}'

            self.heterogeneous_specs  = self.config['heterogeneous_ev_specs']

        self.simulate_grid = False

        self.stats = None

        self.sim_starting_date = self.sim_date

        try:
            with open(self.config['charging_network_topology']) as json_file:
                self.charging_network_topology = json.load(json_file)

        except FileNotFoundError:
            if not self.config['charging_network_topology'] == 'None':
                print("Not file")
            self.charging_network_topology = None

        self.sim_name = extra_sim_name + self.sim_name if extra_sim_name is not None else self.sim_name

        if self.simulate_grid:
            pass
        else:
            if self.charging_network_topology is None:
                self.cs_transformers = [
                    *np.arange(self.number_of_transformers)] * (self.cs // self.number_of_transformers)
                self.cs_transformers += random.sample(
                    [*np.arange(self.number_of_transformers)], self.cs % self.number_of_transformers)
                random.shuffle(self.cs_transformers)

        self.transformers = load_transformers(self)
        for tr in self.transformers:
            tr.reset(step=0)

        self.charging_stations = load_ev_charger_profiles(self) # We will change this formula by our own output or random
        for cs in self.charging_stations:
            cs.reset()

        self.number_of_ports = np.array([cs.n_ports for cs in self.charging_stations]).sum()

        # Load EV spawn scenarios <- It will be changed to our own probabilistic model
        if self.load_from_replay_path is None:
            load_ev_spawn_scenarios(self)

        # Spawn EVs
        self.EVs_profiles = load_ev_profiles(self)
        self.EVs = []

        self.price_data = None
        self.charge_prices, self.discharge_prices = load_electricity_prices(self)

        self.power_setpoints = load_power_setpoints(self)
        self.current_power_usage = np.zeros(self.simulation_length)
        self.charge_power_potential = np.zeros(self.simulation_length)

        self.init_statistic_variables()

        self.done = False

        if self.save_replay:
            os.makedirs(self.replay_path, exist_ok=True)

        if self.render_mode:
            self.renderer = Renderer(self)

        if self.save_plots:
            os.makedirs("./results", exist_ok=True)
            print(f"Creating directory: ./results/{self.sim_name}")
            os.makedirs("./results/{self.sim_name}", exist_ok=True)

        # Action space: is a vector of size "Sum of all ports of all charging stations"
        high = np.ones([self.number_of_ports])
        if self.config['v2g_enabled']:
            lows = -1 * np.ones([self.number_of_ports])
        else:
            lows = np.zeros([self.number_of_ports])

        self.action_space = spaces.Box(low=lows, high=high, dtype=np.float32)

        obs_dim = len(self._get_observation())

        high = np.inf*np.ones([obs_dim])
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.observation_mask = np.zeros(self.number_of_ports)

    def reset(self, seed=None, options=None, **kwargs):
        if seed is None:
            self.seed = np.random.randint(0, 1000000)
        else:
            self.seed = seed

        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.tr_seed == -1:
            self.tr_seed = self.seed
        self.tr_rng = np.random.default_rng(seed=self.tr_seed)

        self.current_step = 0
        self.stats = None

        for cs in self.charging_stations:
            cs.reset()

        for tr in self.transformers:
            tr.reset(step=self.current_step)

        if self.load_from_replay_path is not None or not self.config["random_day"]:
            self.sim_date = self.sim_starting_date
        else:
            if "random_hour" in self.config:
                if self.config['random_hour']:
                    self.config['hour'] = random.randint(5,15)

            self.sim_date = datetime.datetime(2025,
                                              1,
                                              1,
                                              self.config['hour'],
                                              self.config['minute']
                                              ) + datetime.timedelta(days=random.randint(0, int(1.5*365)))

            if self.scenario == 'workplace':
                while self.sim_date.weekday() > 4:
                    self.sim_date += datetime.timedelta(days=1)

            if self.config['simulation_days'] == 'weekdays':
                while self.sim_date.weekday() > 4:
                    self.sim_date += datetime.timedelta(days=1)

            elif self.config['simulation_days'] == 'weekends' and self.scenario != 'workplace':
                while self.sim_date.weekday() <5:
                    self.sim_date += datetime.timedelta(days=1)

        self.sim_starting_date = self.sim_date
        self.EVs_profiles = load_ev_profiles(self)
        self.power_setpoints = load_power_setpoints(self)
        self.EVs = []

        self.init_statistic_variables()

        return self._get_observation(), {}

    def init_statistic_variables(self):
        self.current_step = 0
        self.total_evs_spawned = 0
        self.total_reward = 0

        self.current_ev_departed = 0
        self.current_ev_arrived = 0
        self.current_evs_parked = 0

        self.previous_power_usage = self.current_power_usage
        self.current_power_usage = np.zeros(self.simulation_length)

        self.cs_power = np.zeros([self.cs, self.simulation_length])
        self.cs_current = np.zeros([self.cs, self.simulation_length])

        self.tr_overload = np.zeros([self.number_of_transformers, self.simulation_length])
        self.tr_inflexible_loads = np.zeros([self.number_of_transformers, self.simulation_length])
        self.tr_solar_power = np.zeros([self.number_of_transformers, self.simulation_length])

        if not self.lightweight_plots:
            self.port_current = np.zeros([self.number_of_ports, self.cs, self.simulation_length], dtype=np.float32)
            self.port_current_signal = np.zeros([self.number_of_ports, self.cs, self.simulation_length], dtype=np.float32)
            self.port_energy_level = np.zeros([self.number_of_ports, self.cs, self.simulation_length], dtype=np.float32)
            self.port_arrival = dict({f'{j}.{i}': []
                                    for i in range(self.number_of_ports)
                                    for j in range(self.cs)})

        self.done = False

    def step(self, actions, visualize=False):
        assert not self.done, "Episode is done, Please reset the environment"

        if self.verbose:
            print("-"*80)

        total_costs = 0
        total_invalid_action_punishment = 0
        user_satisfaction_list = [] # whether to fulfill residual SOC in user.
        self.departing_evs = []

        self.current_ev_departed = 0
        self.curent_ev_arrived = 0

        port_counter = 0

        for tr in self.transformers:
            tr.reset(step=self.current_step)

        for i, cs in enumerate(self.charging_stations):
            n_ports = cs.n_ports
            costs, user_satisfaction, invalid_action_punishment, ev = cs.step(
                actions[port_counter:port_counter+n_ports],
                self.charge_prices[cs.id, self.current_step],
                self.discharge_prices[cs.id, self.current_step])

            self.departing_evs += ev

            for u in user_satisfaction:
                user_satisfaction_list.append(u)

            self.current_power_usage[self.current_step] += cs.current_power_output

            self.transformers[cs.connected_transformer].step(
                cs.current_total_amps, cs.current_power_output)

            total_costs += costs
            total_invalid_action_punishment += invalid_action_punishment
            self.current_ev_departed += len(user_satisfaction)

            port_counter += n_ports


        counter = self.total_evs_spawned
        for i, ev in enumerate(self.EVs_profiles[counter:]):
            if ev.time_of_arrival == self.current_step + 1:
                ev = deepcopy(ev)
                ev.reset()
                ev.simulation_length = self.simulation_length
                index = self.charging_stations[ev.location].spawn_ev(ev)

                if not self.lightweight_plots:
                    self.port_arrival[f'{ev.location}.{index}'].append((self.current_step+1, ev.time_of_departure+1))

                self.total_evs_spawned +=1
                self.current_ev_arrived += 1
                self.EVs.append(ev)

            elif ev.time_of_arrival > self.current_step + 1:
                break


        self._update_power_statistics(self.departing_evs)

        self.current_step += 1
        self._step_date()

        if self.current_step < self.simulation_length:
            self.charge_power_potential[self.current_step] = calculate_charge_power_potential(self)

        self.current_evs_parked += self.current_ev_arrived - self.current_ev_departed

        if self.simulate_grid:
            raise NotImplementedError
            grid_report = self.grid.step(actions=actions)
            reward = self._caculate_reward(grid_report)
        else:
            reward = self._caculate_reward(self,
                                           total_costs,
                                           user_satisfaction_list,
                                           total_invalid_action_punishment
                                           )

        if self.cost_function is not None:
            cost = self.cost_function(self,
                                      total_costs,
                                      user_satisfaction_list,
                                      total_invalid_action_punishment)
        else:
            cost = None

        if visualize:
            visualize_step(self)

        self.render()

        return self._check_termination(reward, cost)

    def _check_termination(self, reward, cost):
        truncated = False
        action_mask = np.zeros(self.number_of_ports)
        for i, cs in enumerate(self.charging_stations):
            for j in range(cs.n_ports):
                if cs.evs_connected[j] is not None:
                    action_mask[i*cs.n_ports+j] = 1

        if self.current_step >= self.simulation_length or (any(tr.is_overloaded() > 0 for tr in self.transformers) and not self.generate_rnd_game):
            self.done = True
            self.stats = get_statistics(self)
            self.stats['action_mask'] = action_mask
            self.cost = cost

            if self.verbose:
                print_statistics(self)

                if any(tr.is_overloaded() for tr in self.transformers):
                    print(f"Transformer overloaded, {self.current_step} timesteps\n")
                else:
                    print(f"Episode finished after {self.current_step} timesteps\n")

            if self.save_replay:
                self._save_sim_replay()

            if self.save_plots:
                with open(f"./results/{self.sim_name}/env.pkl", "wb") as f:
                    self.renderer = None
                    pickle.dump(self, f)
                ev_city_plot(self)

            if self.cost_function is not None:
                return self._get_observation(), reward, True, truncated, self.stats
            else:
                return self._get_observation(), reward, True, truncated, self.stats

        else:
            stats = {
                'cost': cost,
                'action_mask': action_mask
            }
            if self.cost_function is not None:
                return self._get_observation(), reward, False, truncated, stats
            else:
                return self._get_observation(), reward, False, truncated, stats

    def render(self):
        if self.render_mode:
            self.renderer.render()

    def _save_sim_replay(self):
        replay = EvCityReplay(self)
        print(f"Saving replay file at {replay.replay_path}")
        with open(replay.replay_path, "wb") as f:
            pickle.dump(replay, f)

        return replay.replay_path

    def set_save_plots(self, save_plots):
        if save_plots:
            os.makedirs(".results", exist_ok=True)
            os.makedirs(f"./results/{self.sim_name}, exist_ok=True")

        self.save_plots = save_plots

    def _update_power_statistics(self, departing_evs):

        for tr in self.transformers:
            self.tr_overload[tr.id, self.current_step] = tr.get_how_overloaded()
            self.tr_inflexible_loads[tr.id, self.current_step] = tr.inflexible_load[self.current_step]
            self.tr_solar_power[tr.id, self.current_step] = tr.solar_power[self.current_step]

        for cs in self.charging_stations:
            self.cs_power[cs.id, self.current_step] = cs.current_power_output
            self.cs_current[cs.id, self.current_step] = cs.current_total_amps

            for port in range(cs.n_ports):
                if not self.lightweight_plots:
                    self.port_current_signal[port, cs.id, self.current_step] = cs.current_signal[port]

                ev = cs.evs_connected[port]
                if ev is not None and not self.lightweight_plots:
                    self.port_current[port, cs.id, self.current_step] = ev.actual_current
                    self.port_energy_level[port, cs.id, self.current_step] = ev.current_capacity / ev.battery_capacity

            for ev in self.departing_evs:
                if not self.lightweight_plots:
                    self.port_energy_level[ev.id, ev.location, self.current_step] = \
                        ev.current_capacity/ev.battery_capacity
                    self.port_current[ev.id, ev.location, self.current_step] = ev.actual_current

    def _step_date(self):
        self.sim_date = self.sim_date + datetime.timedelta(minutes=self.timescale)

    def _get_observation(self):
        return self.state_function(self)

    def set_cost_function(self, cost_function):
        self.cost_function = cost_function

    def set_reward_function(self, reward_function):
        self.reward_function = reward_function

    def _caculate_reward(self, total_costs, user_satisfaction_list, invalid_action_punishment):
        reward = self.reward_function(self, total_costs, user_satisfaction_list, invalid_action_punishment)
        self.total_reward += reward

        return reward







