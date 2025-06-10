import numpy as np
import math

class Transformer():
    def __init__(self,
                 id,
                 env,
                 max_power=100,
                 cs_ids=[],
                 inflexible_load=np.zeros(96),
                 solar_power=np.zeros(96),
                 simulation_length=96
                 ):
        self.id = id
        self.voltage = env.config['charging_station']['voltage'] * math.sqrt(env.config['charging_station']['phases'])
        max_current = max_power * 1000 / self.voltage

        self.max_current = np.ones(simulation_length) * max_current
        self.min_current = np.ones(simulation_length) * -max_current
        self.max_power = np.ones(simulation_length) * max_power
        self.min_power = np.ones(simulation_length) * -max_power

        self.inflexible_load = inflexible_load
        self.solar_power = solar_power

        self.cs_ids = cs_ids
        self.simulation_length = simulation_length

        self.current_amps = 0
        self.current_power = 0

        self.current_step = 0

        self.inflexible_load_forecast = np.zeros(env.simulation_length)
        if env.config['inflexible_loads']['include']:
            self.normalize_inflexible_loads(env)
            self.generate_inflexible_loads_forcast(env)
        self.pv_generation_forecast = np.zeros(env.simulation_length)
        if env.config['solar_power']['include']:
            self.normalize_pv_generation(env)
            self.generate_pv_generation_forecast(env)

        self.steps_ahead = env.config['demand_response']['notification_of_event_minutes'] // env.timescale

        if env.config['demand_response']['include']:
            self.dr_events = self.generate_demand_response_events(env)
            self.inflexible_load_forecast = np.clip(self.inflexible_load_forecast, self.min_power, self.max_power)

        else:
            self.dr_events = []

    def generate_demand_response_events(self, env) -> None:
        events_per_day = env.config['demand_response']['events_per_day']

        event_length_minutes_min = env.config['demand_response']['event_length_minutes_min']
        event_length_minutes_max = env.config['demand_response']['event_length_minutes_max']

        event_start_hour_mean = env.config['demand_response']['event_start_hour_mean']
        event_start_hour_std = env.config['demand_response']['event_start_hour_std']

        event_capacity_percentage_mean = env.config['demand_response']['event_capacity_percentage_mean']
        event_capacity_percentage_std = env.config['demand_response']['event_capacity_percentage_std']

        events = []
        for _ in range(events_per_day):
            event_length_minutes = env.tr_rng.integers(event_length_minutes_min, event_length_minutes_max+1)

            event_start_hour = env.tr_rng.normal(event_start_hour_mean * 60, event_start_hour_std * 60)

            event_start_hour = np.clip(event_start_hour, 0, 23*60)
            event_start_step = event_start_hour / env.timescale

            sim_start_step = (env.sim_date.hour * 60 + env.sim_date.minute) //env.timescale

            event_start_step = int(event_start_step - sim_start_step)

            event_end_step = int(event_start_step + event_length_minutes // env.timescale)

            capacity_percentage = env.tr_rng.normal(event_capacity_percentage_mean, event_capacity_percentage_std)
            capacity_percentage = np.clip(capacity_percentage, 0, 100)

            self.max_power[event_start_step:event_end_step] = self.max_power[event_start_step:event_end_step] - (self.max_power[event_start_step:event_end_step] * capacity_percentage/100)
            self.max_current[event_start_step:event_end_step] = self.max_current[event_start_step:event_end_step] - (self.max_current[event_start_step:event_end_step] * capacity_percentage/100)

            if any(self.inflexible_load[event_start_step:event_end_step] > self.max_power[event_start_step:event_end_step]):
                self.max_power[event_start_step:event_end_step] = self.inflexible_load[event_start_step:event_end_step].max()

                capacity_percentage = 100 * (1 - max(self.inflexible_load[event_start_step:event_end_step]) / max(self.max_power))

            event = {'event_start_step': event_start_step,
                     'event_end_step': event_end_step,
                     'capacity_percentage': capacity_percentage}
            events.append(event)

        return events
