import os
import pandas as pd
import numpy as np
from scipy import interpolate
import math

class EVs:
    def __init__(self, **kwargs):
        kwargs = {k.lower(): v for k, v in kwargs.items()}
        '''
        0. it is a dict comprehension used to create a new dict from kwargs
        1. key(k), value (v)
        2. For each key-value pair, the key (k) is converted to lowercase using the .lower().
        '''

        if 'vehicle_type' in kwargs:
            self.evtype = kwargs['vehicle_type']
        else:
            self.evtype = 'bev'

        self.arrivaltime = kwargs['arrival_time']
        self.initialalsoc = kwargs['initial_soc']
        self.modelparameters = self.load_ev_file(**kwargs)

        if 'batterycapacity_kwh' in kwargs:
            self.modelparameters['ev_packcapacity'] = kwargs['batterycapacity_kwh']

        self.soc = self.initialalsoc
        self.timestep_soc = self.arrivaltime
        self.packvoltage = self.getocv(self.soc, **kwargs)[0]
        self.packpower = 0.0
        self.packcurrent = 0.0
        self.pluggedin = False
        self.readytocharge = False
        self.chargecomplete = False
        if 'target_soc' in kwargs:
            self.targetsoc = kwargs['target_soc']
        else:
            self.targetsoc = 1.0

        if 'departure_time' in kwargs:
            self.departuretime = kwargs['departure_time']
        else:
            self.departuretime = self.arrivaltime + 24.0 * 3600
        self.evse_id = np.nan

    def isvehiclepluggedin(self, simulationtime):
        if (simulationtime >= self.arrivaltime) and (simulationtime < self.arrivaltime + self.modelparameters['ev_setuptime']) and (simulationtime <= self.departuretime):
            self.pluggedin = True
            self.readytocharge = False

        elif (simulationtime >= self.arrivaltime + self.modelparameters['ev_setuptime']) and (simulationtime <= self.departuretime):
            self.pluggedin = True
            self.readytocharge = True
        else:
            self.pluggedin = False
            self.readytocharge = False


    def assign_evse(self, evse_id):
        self.evse_id = evse_id

    def ischargecomplete(self, simulationtime, **kwargs):
        if simulationtime >= self.departuretime or self.soc >= self.targetsoc:
            self.chargecomplete = True
        else:
            self.chargecomplete = False

    def load_ev_file(self, **kwargs):

