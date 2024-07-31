import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt
from keychain import Keychain as kc


params = open(kc.INPUT_FILE)
params = json.load(params)

config_params = params[kc.CONFIG]
data_params = params[kc.PARAMS_DATA]
experiment_params = params[kc.EDATA]
datadirpath = config_params["path" + config_params[kc.WHICH_PATH]]

data_params = params[kc.PARAMS_DATA]

RoadNetworks = data_params[kc.ROAD_NETWORKS]
NetworkDayChange = data_params[kc.NETWORK_DAY_CHANGE]

net_params = params[RoadNetworks[0]]

EDIRNAME = experiment_params[kc.NAME] + experiment_params[kc.Xaxis] + experiment_params[kc.Yaxis]
dirpath = datadirpath + "./" + EDIRNAME + "/"

NR = int(net_params[kc.ROUTES][kc.NUMBER_OF_ROUTES])
RouteLabels = net_params[kc.ROUTE_LABELS]

EFILENAME = experiment_params[kc.NAME]
EXaxis = experiment_params[kc.Xaxis]
EYaxis = experiment_params[kc.Yaxis]
EYval = [(x) for x in(experiment_params[kc.Yval])]
EXval = [(x) for x in(experiment_params[kc.Xval])]
EYvalN = len(EYval)
EXvalN = len(EXval)
EXaxisName = experiment_params[kc.XaxisName]
EYaxisName = experiment_params[kc.YaxisName]
plotCAVs = data_params[kc.PLOT_CAVS]
plotMeanHDV = data_params[kc.PLOT_MEAN_HDV] 

fleetIntro = data_params[kc.DEFAULT_FLEET_INTRODUCTION]
print("Experimental data loaded")
dfarray = {}
for exind in range(EXvalN):
    for eyind in range(EYvalN):
        filepath = dirpath + EFILENAME + str('_')+ str(EXval[exind]) + str('_') + str(EYval[eyind]) + '.csv'
        try:
            dfarray[exind, eyind] = pd.read_csv(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Please run BottleCOEX.py before running this script. File {filepath} not found.")


jrange = data_params[kc.DAY_RANGE]
Labels_counts = ["HDV count on A", "HDV count on B", "CAV count on A", "CAV count on B"]
Labels_countsTT = ["Travel time on A", "Travel time on B", "Mean HDV travel time", "Mean CAV travel time"]
            
################## plot0A ############

colorNames = ["blue", "orange", "green", "red"]
superior_title_end = ''
superior_title_end2 = ''
for r in range(NR):
    superior_title_end = superior_title_end + RouteLabels[r] + " (" + colorNames[r] + ")"
    if(r != (NR-1)): superior_title_end = superior_title_end + " and "
for r in range(NR):
    superior_title_end2 = superior_title_end2 + RouteLabels[r] + " (" + colorNames[NR+r] + ") "
    if(r != (NR-1)): superior_title_end2 = superior_title_end2 + "and "




fig = plt.figure()
gs = fig.add_gridspec(EYvalN, EXvalN, hspace=0, wspace=0)
axs = gs.subplots(sharex=True, sharey=True)
    
superior_title = 'Travel times on ' + superior_title_end
if(plotMeanHDV): superior_title = superior_title+ ', mean for HDV (' + colorNames[NR] + ')'
if(plotCAVs): superior_title = superior_title + ' and CAV (' + colorNames[NR+1] + ')'
fig.suptitle(superior_title)
fig.supxlabel('Day')
fig.supylabel('Time [min]')
for exind in range(EXvalN):
    for eyind in range(EYvalN):
        ax = axs[eyind, exind]
        for r in range(NR):
            ax.plot(dfarray[exind, eyind]["Travel time on " + RouteLabels[r]])
        if(plotMeanHDV): ax.plot(dfarray[exind, eyind]["Mean HDV travel time"])
        if(plotCAVs):
            ax.plot(dfarray[exind, eyind]["Mean CAV travel time"][data_params["defaultFleetIntroduction"]:data_params["dayRange"]])
            ax.axvline(x=200, ymin=0, ymax = 0.8, color = 'black', linestyle = 'dashed', linewidth = 0.5)

    
        if(dfarray[exind, eyind]["Mean HDV Perceived travel time"][1] > 0):
            ax.plot(dfarray[exind, eyind]["Mean HDV Perceived travel time"])

        ax.text(0.1, 0.95, EXaxisName + " = " + str(EXval[exind]) + ", " + EYaxisName + " = " + str(EYval[eyind]),
                transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')
        ax.minorticks_on()
        ax.grid(axis = 'y', which='minor', color='#EEEEEE', linestyle=(0,(1,1)), linewidth=0.6)
        ax.grid(axis = 'y')
plt.show()

################## plot0B ############

fig = plt.figure()
gs = fig.add_gridspec(EYvalN, EXvalN, hspace=0, wspace=0)
axs = gs.subplots(sharex=True, sharey=True)
    
superior_title = 'HDVs on ' + superior_title_end
if(plotCAVs): superior_title = superior_title + 'and CAVs on ' + superior_title_end2

fig.suptitle(superior_title)
fig.supxlabel('Day')
fig.supylabel('Vehicle count')
for exind in range(EXvalN):
    for eyind in range(EYvalN):
        ax = axs[eyind, exind]
        for r in range(NR):
            ax.plot(dfarray[exind, eyind]["HDV count on " + RouteLabels[r]])
        if(plotCAVs):
            for r in range(NR):
                ax.plot(dfarray[exind, eyind]["CAV count on " + RouteLabels[r]][data_params[kc. DEFAULT_FLEET_INTRODUCTION]:data_params[kc.DAY_RANGE]])
            ax.axvline(x=200, ymin=0, ymax = 0.8, color = 'black', linestyle = 'dashed', linewidth = 0.5)
        ax.text(0.1, 0.95, EXaxisName + " = " + str(EXval[exind]) + ", " + EYaxisName + " = " + str(EYval[eyind]),
                transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')
        ax.minorticks_on()
        ax.grid(axis = 'y', which='minor', color='#EEEEEE', linestyle=(0,(1,1)), linewidth=0.6)
        ax.grid(axis = 'y')
plt.show()

