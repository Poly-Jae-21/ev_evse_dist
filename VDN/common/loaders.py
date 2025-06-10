from typing import List, Tuple

import numpy as np


def load_transformers(env) -> List[Transformer]:
    if env.load_from_replay_path is not None:
        return env.replay.transformers

    transformers = []

    if env.config['inflexible_loads']['include']:
        if env.scenario == 'private':
            inflexible_loads = generate_residential_inflexible_loads(env)

        else:
            inflexible_loads = generate_residential_inflexible_loads(env)

    else:
        inflexible_loads = np.zeros((env.number_of_transformers, env.simulation_length))

    if env.config['solar_power']['include']:
        solar_power = generate_pv_generation(env)
    else:
        solar_power = np.zeros((env.number_of_transformers, env.simulation_length))

    if env.charging_network_topology:
        cs_counter = 0
        for i, tr in enumerate(env.charging_network_topology):
            cs_ids = []
            for cs in env.charging_network_topology[tr]['charging_stations']:
                cs_ids.append(cs_counter)
                cs_counter += 1
            transformer = Transformer(id=i,
                                      env=env,
                                      cs_ids=cs_ids,
                                      max_power=env.charging_network_topology[tr]['max_power'],
                                      inflexible_loads=inflexible_loads[i, :],
                                      solar_power=solar_power[i, :],
                                      simulation_length=env.simulation_length
                                      )

            transformers.append(transformer)

    else:
        if env.number_of_transformers > env.cs:
            raise ValueError("The number of transformers cannot exceed the number of charging stations.")
        for i in range(env.number_of_transformers):
            transformer = Transformer(id=i,
                                      env=env,
                                      cs_ids=np.where(np.array(env.cs_transformers)==i)[0],
                                      max_power=env.config['transformer']['max_power'],
                                      inflexible_loads=inflexible_loads[i, :],
                                      solar_power=solar_power[i, :],
                                      simulation_length=env.simulation_length
                                      )
            transformers.append(transformer)

    env.n_transformers = len(transformers)
    return transformers