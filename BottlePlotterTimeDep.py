import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt

dirpath=""
fE = open(dirpath + 'EData.json')
dataE = json.load(fE)
fE.close()
f = open(dirpath + 'paramsData.json')
data = json.load(f)
f.close()
fN = open(dirpath + data["RoadNetworks"][0] + ".json")
dataN = json.load(fN)
fN.close()

EDIRNAME = dataE["name"] + dataE["Xaxis"] + dataE["Yaxis"]
dirpath = "./" + EDIRNAME + "/"

NR = int(dataN["Routes"]["NumberOfRoutes"])
RouteLabels = dataN["RouteLabels"]

EFILENAME = dataE["name"]
EXaxis = dataE["Xaxis"]
EYaxis = dataE["Yaxis"]
EYval = [(x) for x in(dataE["Yval"])]
EXval = [(x) for x in(dataE["Xval"])]
EYvalN = len(EYval)
EXvalN = len(EXval)
EXaxisName = dataE["XaxisName"]
EYaxisName = dataE["YaxisName"]
plotCAVs = data["PlotCAVs"]
plotMeanHDV = data["PlotMeanHDV"] 

fleetIntro = data["defaultFleetIntroduction"]
print("Experimental data loaded")
dfarray = {}
for exind in range(EXvalN):
    for eyind in range(EYvalN):
        filepath = dirpath + EFILENAME + str('_')+ str(EXval[exind]) + str('_') + str(EYval[eyind]) + '.csv'
        dfarray[exind, eyind] = pd.read_csv(filepath)


jrange = data["dayRange"]
Labels_counts = ["HDV count on A", "HDV count on B", "CAV count on A", "CAV count on B"]
Labels_countsTT = ["Travel time on A", "Travel time on B", "Mean HDV travel time", "Mean CAV travel time"]
            
##################plot0A############

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
            ax.plot(dfarray[exind, eyind]["Mean CAV travel time"][data["defaultFleetIntroduction"]:data["dayRange"]])
            ax.axvline(x=200, ymin=0, ymax = 0.8, color = 'black', linestyle = 'dashed', linewidth = 0.5)

    
        if(dfarray[exind, eyind]["Mean HDV Perceived travel time"][1] > 0):
            ax.plot(dfarray[exind, eyind]["Mean HDV Perceived travel time"])

        ax.text(0.1, 0.95, EXaxisName + " = " + str(EXval[exind]) + ", " + EYaxisName + " = " + str(EYval[eyind]),
                transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')
        ax.minorticks_on()
        ax.grid(axis = 'y', which='minor', color='#EEEEEE', linestyle=(0,(1,1)), linewidth=0.6)
        ax.grid(axis = 'y')
plt.show()

##################plot0B############

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
                ax.plot(dfarray[exind, eyind]["CAV count on " + RouteLabels[r]][data["defaultFleetIntroduction"]:data["dayRange"]])
            ax.axvline(x=200, ymin=0, ymax = 0.8, color = 'black', linestyle = 'dashed', linewidth = 0.5)
        ax.text(0.1, 0.95, EXaxisName + " = " + str(EXval[exind]) + ", " + EYaxisName + " = " + str(EYval[eyind]),
                transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')
        ax.minorticks_on()
        ax.grid(axis = 'y', which='minor', color='#EEEEEE', linestyle=(0,(1,1)), linewidth=0.6)
        ax.grid(axis = 'y')
plt.show()

