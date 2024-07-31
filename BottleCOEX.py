import os
import csv
import json
import time
import copy
import shutil
import numpy as np
import pandas as pd
from numpy import random
from matplotlib import pyplot as plt

from fleet_operator_agent import FleetOperatorAgent
from keychain import Keychain as kc
from route import Route
from utils import routesCounts2linksCounts
from create_network import createNetwork
from human_agent import HumanAgent
from simulator import Simulator



if __name__ == "__main__":

    ## Initialize parameters
    params = open(kc.INPUT_FILE)
    params = json.load(params)

    params_data = params[kc.PARAMS_DATA]
    experiment_data = params[kc.EDATA]
    human_params = params[kc.HUMAN_PARAMS]
    config_params = params[kc.CONFIG]

    RoadNetworks = params_data[kc.ROAD_NETWORKS]
    NetworkDayChange = params_data[kc.NETWORK_DAY_CHANGE]

    net_params = params[RoadNetworks[0]]
    datadirpath = config_params["path" + config_params[kc.WHICH_PATH]]

    ENAME = experiment_data[kc.NAME]
    EXaxis = experiment_data[kc.Xaxis]
    EYaxis = experiment_data[kc.Yaxis]
    EDIRNAME = (datadirpath + ENAME + EXaxis + EYaxis)
    EXval = [(x) for x in(experiment_data[kc.Xval])]
    EYval = [(x) for x in(experiment_data[kc.Yval])]
    EXvalN = len(EXval)
    EYvalN = len(EYval) 
    EXaxisName = experiment_data[kc.XaxisName]
    EYaxisName = experiment_data[kc.YaxisName]

    ## Create the network
    NL, Links = createNetwork(net_params)

    for l in range(NL):
        print("Link " + str(l))
        Links[l].print()

    ## Create the routes    
    Routes = []
    NR = int(net_params[kc.ROUTES][kc.NUMBER_OF_ROUTES]) 
    for r in range(NR):
        Routes.append(Route(Links, net_params[kc.ROUTES][kc.LINKS_MATRIX][r], net_params[kc.ROUTE_LABELS][r]))

    ## Create the experiment directory
    try:
        os.mkdir(EDIRNAME)

    except FileExistsError:
        # If the directory exists, delete its contents
        shutil.rmtree(EDIRNAME)
        os.mkdir(EDIRNAME)

    output_file = EDIRNAME + "/" + kc.EDATA +'.json'
    with open(output_file, 'w') as json_file:
        json.dump(experiment_data, json_file, indent=4)

    output_file = EDIRNAME + "/" + kc.PARAMS_DATA +'.json'
    with open(output_file, 'w') as json_file:
        json.dump(params_data, json_file, indent=4)

    
    for rn in range(len(RoadNetworks)): 
        output_file = EDIRNAME + "/" + params_data[kc.ROAD_NETWORKS][rn] + '.json'

        with open(output_file, 'w') as json_file:
            json.dump(params[params_data[kc.ROAD_NETWORKS][rn]], json_file, indent=4)
        
        
    print("Experimental data loaded")
    print(str(EYaxis) + " used in experiments:" + str(EYval))
    print(str(EXaxis) + " used in experiments:" + str(EXval))

    ## Main experiment loop
    for exind in range(EXvalN):
        for eyind in range(EYvalN):

            d = {EXaxis: EXval[exind], EYaxis: EYval[eyind]}
            print(d)                

            (TTdfs, VCdfs) = Simulator(RoadNetworks, NetworkDayChange, NL, Links, NR, Routes, human_params, params_data, **d)

            ToFiledf = TTdfs[["Day",] + ["Travel time on " + Routes[r].label for r in range(NR)] +
                                ["Mean HDV travel time",] + ["Mean HDV Perceived travel time",] + ["Mean CAV travel time",]].copy()
            
            ToFiledf2 = VCdfs[["Day",] + ["HDV count on " + Routes[r].label for r in range(NR)]
                                        + ["CAV count on " + Routes[r].label for r in range(NR)]].copy()
            
            filepath = "./" + EDIRNAME + "/" + ENAME + str('_')+ str(EXval[exind]) + str('_') + str(EYval[eyind]) + '.csv'
            ToFiledf.merge(ToFiledf2, how='left', on='Day').to_csv(filepath)
            
            print(EXaxisName + ": " + str(EXval[exind]) + " " + EYaxisName + ": " + str(EYval[eyind]) + " Done")    

####################

