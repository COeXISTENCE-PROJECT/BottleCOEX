import copy
import json
import numpy as np
from numpy import random
import pandas as pd
import random

from .create_network import createNetwork
from .human_agent import HumanAgent
from keychain import Keychain as kc
from .fleet_operator_agent import FleetOperatorAgent
from .utils import routesCounts2linksCounts



params = open(kc.INPUT_FILE)
params = json.load(params)

config_params = params[kc.CONFIG]
data_params = params[kc.PARAMS_DATA]
datadirpath = config_params["path" + config_params[kc.WHICH_PATH]]
  
DEFAULT_TOTALVEHICLES = (float(data_params[kc.TOTAL_VEHICLES]))

mult = data_params[kc.MULTIPLIER]                                              
DEFAULT_TOTALND = int(DEFAULT_TOTALVEHICLES)                                                 

#####The main Simulator, computes one scenario with with fixed parameters#####
def Simulator(RoadNetworks, NetworkDayChange, NL, Links, NR, Routes, human_params, params_data, Fleet_Mode = data_params[kc.DEFAULT_FLEET_MODE], 
              Model_Name = data_params[kc.DEFAULT_MODEL_NAME], Alpha_Zero = data_params[kc.HDVS][kc.DEFAULT_ALPHA_ZERO],
              Epsilon_Zero =  data_params[kc.HDVS][kc.DEFAULT_EPSILON], Logit_Exp_Coeff = data_params[kc.DEFAULT_LOGIT_PARAM],
              Initial_Knowledge = data_params[kc.DEFAULT_INITIAL_KNOWLEDGE], Initial_Choice = data_params[kc.DEFAULT_INITIAL_CHOICE], 
              Only_Experience = data_params[kc.DEFAULT_ONLY_EXPERIENCE], Fleet_size = int(float(data_params[kc.DEFAULT_FLEET_SIZE])),
              Fleet_introduction = data_params[kc.DEFAULT_FLEET_INTRODUCTION], LambdaHDV = float(data_params[kc.DEFAULT_LAMBDA_HDV]), 
              LambdaCAV = float(data_params[kc.DEFAULT_LAMBDA_CAV]), CAVTarget = (data_params[kc.DEFAULT_CAV_TARGET]), 
              jrange = data_params[kc.DAY_RANGE], totalND = int(float(data_params[kc.TOTAL_VEHICLES]))):

    #####Normalize variables by mult
    totalND = int(mult*(totalND))
    Fleet_size = int((totalND) * Fleet_size)
    #####Create human drivers########
    HDV = []
    Qinit = np.zeros(NR)
    for i in range(totalND):
        INITIAL = Initial_Knowledge
        INITIALCHOICE = Initial_Choice
        for r in range(NR):  
            if(INITIAL == 'FreeFlow'):      Qinit[r] = Routes[r].t0
            elif(INITIAL == 'Pessimistic'): Qinit[r] = 25.0
            elif(INITIAL == 'Optimistic'):  Qinit[r] = 0.0

        HDV.append(HumanAgent(human_params, jrange, Model_Name, Alpha_Zero, human_params[kc.LEARNING_RATE_DECAY], Epsilon_Zero,
                              human_params[kc.EXPLORATION_RATE_DECAY], Qinit, INITIALCHOICE, Logit_Exp_Coeff, Only_Experience, NR))
        
    ####Create fleet agent
    if(CAVTarget == 'Selfish'): LambdaHDV, LambdaCAV = 0.0, 1.0 
    elif(CAVTarget == 'Malicious'): LambdaHDV, LambdaCAV = -1.0, 0.0 
    elif(CAVTarget == 'Disruptive'): LambdaHDV, LambdaCAV = -9.0, 1.0 
    elif(CAVTarget == 'Social'): LambdaHDV, LambdaCAV = 1.0, 1.0 
    elif(CAVTarget == 'Altruistic'): LambdaHDV, LambdaCAV = 1.0, 0.0 

    F1 = FleetOperatorAgent(NR, Routes, NL, Links, Fleet_size, Fleet_introduction, LambdaHDV, LambdaCAV, Fleet_Mode)        
    ####Initialize other variables
    averagingSampleSize = totalND - F1.FS
    routeTimes = np.zeros((jrange,NR))                                
    averageHDVTravelTime = np.zeros(jrange)                            
    averageCAVTravelTime = np.zeros(jrange)                           
    averagePerceivedTravelTime = np.zeros(jrange)                      
    averageTravelTime = np.zeros(jrange)
    storedTarget = np.zeros(jrange)
    CumqcountsHDV = np.zeros((jrange, NR), dtype=float)                 
    noChangeRoutes = np.zeros(jrange)               
    qcountsHDV = np.zeros((jrange, NR), dtype=int)                      
    qcountsCAV = np.zeros((jrange, NR), dtype=int)                      
    for i in range(totalND):
        qcountsHDV[0, HDV[i].curRoute] += 1

    ND = np.zeros(jrange, dtype = int)
    ND[0] = totalND  #we assume that there is no fleet on day 0
    curNetwork = 0
    noDriversChangingRoutes = 0                                                                       

    #########Main loop over days############
    random.seed()
    for j in range(jrange):

        #Evaluate route times on day j
        linksCounts = routesCounts2linksCounts(NR, Routes, NL, Links, (qcountsHDV[j, :] + qcountsCAV[j, :]))
        for r in range(NR):
            routeTimes[j, r] = Routes[r].travel_time(Links, linksCounts)
            routeTimes[j, r] += np.random.normal(loc=0.0, scale=human_params[kc.ROUTE_RANDOM_VAR])

        #Save statistics for day j
        noChangeRoutes[j] = noDriversChangingRoutes                                                  
        if(ND[j]>0):
            averageHDVTravelTime[j] = np.sum(routeTimes[j, :] * qcountsHDV[j, :])/ND[j] 

        if(F1.during_optimization(j)):
            CAVTotal = F1.FS
            if(CAVTotal > 0):
                averageCAVTravelTime[j] = np.sum(routeTimes[j, :]*qcountsCAV[j,:]) / CAVTotal
            storedTarget[j] = (F1.Target((ND[j]*averageHDVTravelTime[j]), (CAVTotal*averageCAVTravelTime[j])))/(totalND)
            averageTravelTime[j] = (np.sum(routeTimes[j, :]*qcountsHDV[j, :]) + np.sum(routeTimes[j, :]*qcountsCAV[j,:]))/totalND
        else:
            averageTravelTime[j] = averageHDVTravelTime[j]

        if((Model_Name == 'Gumbel') or (Model_Name == 'GumbelEps')):
            averagePerceivedTravelTime[j] = 0                                                            
            #Perceived travel time only for HDVs which do not mutate into CAVs
            for i in range(averagingSampleSize):
                averagePerceivedTravelTime[j] += (routeTimes[j, HDV[i].curRoute] + HDV[i].gumbelOffsets[HDV[i].curRoute])  
            if (averagingSampleSize > 0):
                averagePerceivedTravelTime[j] = averagePerceivedTravelTime[j] / (averagingSampleSize)         
        else:
            averagePerceivedTravelTime[j] = -1

        #Don't learn or make future choices if it's the last day
        if((j==(jrange-1))): break    

        #Update number of human drivers ND (the last F1.FS become fleet vehicles)
        if(F1.during_optimization(j+1)): ND[j+1] = totalND - F1.FS                                                                
        else: ND[j+1] = totalND

        #Update network
        try:
            if(j == NetworkDayChange[curNetwork]):
                curNetwork += 1
                fN = open(datadirpath + RoadNetworks[curNetwork] + ".json")
                net_params = json.load(fN)
                fN.close()
                NL, Links = createNetwork(net_params)
        except:
            raise(Exception("Network update failed"))

        #######Main iteration over drivers - learning and route choice###################################
        noDriversChangingRoutes = 0                                                                       
        OLDHDV = copy.deepcopy(HDV)
        for i in range(ND[j+1]):                                                                 
            HDV[i].learn(routeTimes[j,:])
            HDV[i].update_rates()                                                          
            oldRoute = HDV[i].curRoute                                                      
            HDV[i].make_next_route_choice()
            qcountsHDV[j+1, HDV[i].curRoute] += 1
            if(oldRoute != HDV[i].curRoute): noDriversChangingRoutes += 1
        ######end of iteration over drivers, optimization by Fleet
        NEWHDV = copy.deepcopy(HDV)
        F1.make_next_routes_choice(j+1, Routes, Links, NEWHDV, ND[j+1], qcountsHDV[j+1], qcountsHDV[j], routeTimes[j,:], OLDHDV) 
        qcountsCAV[j+1, :] = F1.Intercepts[:]
    ######End of main loop over days

    ####Process data for output
    convVector = np.ones(params_data["MovingAverageStep"]) / params_data["MovingAverageStep"]


    for r in range(NR):
        qcI = qcountsHDV[:,r]
        convol = np.convolve(qcI, convVector)
        for j in range(jrange):
            CumqcountsHDV[j,r] = convol[j]
    CumaverageCAVTravelTime = np.convolve(averageCAVTravelTime, convVector)
    CumaverageTravelTime = np.convolve(averageHDVTravelTime, convVector)
    CumaveragePerceivedTravelTime = np.convolve(averagePerceivedTravelTime, convVector)
    MovAvstoredTarget = np.convolve(storedTarget, convVector)
    CumstoredTarget = np.zeros(jrange)
    OptMaxIndex = jrange
    OptimizationOnset = Fleet_introduction
    CumstoredTarget[OptimizationOnset:OptMaxIndex] = storedTarget[OptimizationOnset:OptMaxIndex] / (OptMaxIndex-OptimizationOnset)
    for j in range(OptimizationOnset, OptMaxIndex): CumstoredTarget[j] += CumstoredTarget[j-1]


    TTdf = pd.DataFrame(data = np.concatenate((np.arange(jrange)[:,None],routeTimes,storedTarget[:,None],
                                               CumstoredTarget[:,None],averageHDVTravelTime[:,None],
                                               averagePerceivedTravelTime[:,None],averageCAVTravelTime[:,None],
                                               averageTravelTime[:,None],
                                               CumaverageTravelTime[:jrange,None],
                                               CumaverageCAVTravelTime[:jrange,None],
                                               CumaveragePerceivedTravelTime[:jrange,None],),axis=1),
                        columns = ["Day",] + ["Travel time on " + Routes[r].label for r in range(NR)]
                                           + ["Target", "CumulativeTarget", "Mean HDV travel time",
                                              "Mean HDV Perceived travel time", "Mean CAV travel time",
                                              "Mean travel time of all vehicles", "Moving Average Mean HDV travel time",
                                              "Moving Average Mean CAV travel time", "Moving Average Mean HDV Perceived travel time"])
    VCdf = pd.DataFrame(data = np.concatenate((np.arange(jrange)[:,None],qcountsHDV, CumqcountsHDV, qcountsCAV,noChangeRoutes[:,None]
                                               ), axis=1),
                        columns = ["Day",] + ["HDV count on " + Routes[r].label for r in range(NR)]
                                           + ["MovAveHDV count on " + Routes[r].label for r in range(NR)]
                                           + ["CAV count on " + Routes[r].label for r in range(NR)]
                                           + ["Number of HDVs changing Routes",]
                                            )
        
    return ((TTdf, VCdf))

