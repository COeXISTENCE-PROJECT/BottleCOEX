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

from keychain import Keychain as kc


params = open(kc.INPUT_FILE)
params = json.load(params)

config_params = params[kc.CONFIG]
data_params = params[kc.PARAMS_DATA]
datadirpath = config_params["path" + config_params["whichPath"]]


"""fC = open("Config.json")
dataC = json.load(fC)
fC.close()
datadirpath =  dataC[("path" + dataC["whichPath"])]
print("\n\ndatadirpath is: ", datadirpath, "\n\n\n")"""

inputFileName = 'paramsData'
f = open(datadirpath + inputFileName + '.json')
paramsData = json.load(f)
f.close()

DEFAULT_ALPHA_ZERO = paramsData["HDVs"]["defaultAlphaZero"]
DEFAULT_EPSILON_ZERO = paramsData["HDVs"]["defaultEpsilon"]
DEFAULT_LOGIT_PARAM = paramsData["defaultLogitParam"]
DEFAULT_INITIAL_KNOWLEDGE = paramsData["defaultInitialKnowledge"]
DEFAULT_INITIAL_CHOICE = paramsData["defaultInitialChoice"]
DEFAULT_ONLY_EXPERIENCE = paramsData["defaultOnlyExperience"]
DEFAULT_MODEL_NAME = paramsData["defaultModelName"]
DEFAULT_JRANGE = paramsData["dayRange"]    
DEFAULT_TOTALVEHICLES = (float(paramsData["totalVehicles"]))

mult = paramsData["multiplier"]                                              
DEFAULT_TOTALND = int(DEFAULT_TOTALVEHICLES)                                                 


DEFAULT_LAMBDA_CAV = float(paramsData["defaultLambdaCAV"])                              
DEFAULT_LAMBDA_HDV = float(paramsData["defaultLambdaHDV"])                              
DEFAULT_CAV_TARGET = (paramsData["defaultCAVTarget"])                              
DEFAULT_FLEET_SIZE = int(float(paramsData["defaultFleetSize"]))
DEFAULT_FLEET_INTRODUCTION = paramsData["defaultFleetIntroduction"]
DEFAULT_FLEET_MODE = paramsData["defaultFleetMode"]


ROUTE_RANDOM_VAR = 0.0
DRIVER_RANDOM_VAR = 0.0
LEARNING = True
LEARNING_RATE_DECAY = 1.0
EXPLORATION_RATE_DECAY = 1.0


#############Classes definitions (Link, Route, HumanAgent, Mode, FleetOperatorAgent)####################
class Link:

    def __init__(self, t0, kappa, cap, power):
        self.t0 = t0
        self.kappa = kappa
        self.cap = cap
        self.power = power

    def print(self):
        print("t0:" + str(self.t0) + " cap:" + str(self.cap) + " kappa:" + str(self.kappa) + " power:" + str(self.power))

    def travel_time(self, count):
        return BPR(self.t0, self.kappa, self.cap, self.power, count)

class Route:

    def __init__(self, links, link_incidency, label):
        self.link_incidency = link_incidency
        self.label = label
        self.t0 = self.travel_time(links, np.zeros(len(links)))
        
    def travel_time(self, links, counts_on_links):
        tt = 0
        for l in range(len(self.link_incidency)):
            if(self.link_incidency[l] > 0):
                tt = tt + links[l].travel_time(counts_on_links[l])
        return tt
        
class HumanAgent:

    def __init__(self, jrange, model_name, learning_rate, learning_rate_decay, exploration_rate, exploration_rate_decay,
                 Qinit, Qinitchoice, logit_par, only_experience, NR):
        self.NR = NR
        self.lr = learning_rate
        self.er = exploration_rate
        self.initlr = self.lr
        self.initer = self.er
        self.lr_decay = learning_rate_decay
        self.er_decay = exploration_rate_decay
        self.Q = np.zeros(self.NR)
        self.Q[:] = Qinit[:]
        self.Qstored = np.zeros((jrange, self.NR))
        if(Qinitchoice == 'Argmin'):  self.curRoute = np.argmin(self.Q[:])
        elif(Qinitchoice == 'Random'):self.curRoute = random.randint(self.NR)
        else: raise Exception("Initial choice not specified")
        self.minRoute = 0
        self.learning_ctr = 0
        self.learns = LEARNING
        self.learn_from_experience = only_experience
        self.logit_par = logit_par
        gumbelScale = 1/self.logit_par  
        gumbelMean = 0
        gumbelLocation = gumbelMean - (gumbelScale*np.euler_gamma)            #location of Gumbel can be computed using the scale and mean only (and Euler-Mascheroni constant)
        self.gumbelOffsets = np.random.gumbel(gumbelLocation, gumbelScale, self.NR)        
        self.model_name = model_name
        if(self.model_name == "Gumbel"): self.er = 0

    def learn(self, t_routes):
        self.Qstored[self.learning_ctr,:] = self.Q[:]
        self.learning_ctr += 1
        if(self.learns):
            if(self.learn_from_experience):
                self.Q[self.curRoute] = (1-self.lr) * self.Q[self.curRoute] + (self.lr*(t_routes[self.curRoute]
                                                                                     + random.normal(loc=0.0, scale=DRIVER_RANDOM_VAR)))
            else:
                self.Q[:] = (1-self.lr)*self.Q[:] + self.lr*(t_routes[:]) 
            if self.Q[self.curRoute] < self.Q[self.minRoute]:
                self.minRoute = self.curRoute

    def update_rates(self):
        self.er *= self.er_decay
        self.lr *= self.lr_decay

    def make_next_route_choice(self):
        if(self.model_name == "Gumbel"):
            self.make_Gumbel_choice(random.random())
        elif(self.model_name == "GumbelEps"):
            self.make_Gumbel_choice(random.random())
        elif(self.model_name == "Logit"):
            self.make_Logit_choice(random.random())                                   
        elif(self.model_name == "EpsGreedy"):
            self.make_epsilon_greedy_choice(random.random())
        else: raise Exception("Choice model not specified")
        
    def make_epsilon_greedy_choice(self, random_value):
        minindexEG = np.random.choice(np.flatnonzero(self.Q[:] == self.Q[:].min()))                                         
        if (random_value < self.er): self.curRoute = random.randint(self.NR)
        else: self.curRoute = minindexEG
        
    def make_Gumbel_choice(self, random_value):
        QG = self.Q[:] + self.gumbelOffsets[:]                                          
        self.curRoute = eps_greedy(self.er, random_value, self.NR, np.argmin(QG[:]))         

    def make_Logit_choice(self, random_value):
        exps = np.exp(-self.logit_par*self.Q[:])
        denominator = np.sum(exps)
        epsilons = exps / denominator
        r=0
        threshold = 0.0
        while(r < self.NR):
            threshold += epsilons[r]
            if(random_value < threshold):
                self.curRoute = r
                r = self.NR
            r = r+1

class Mode:
    
    def __init__(self, n_steps_exact, n_steps_model, n_runs = 1, mode_name = ''):
        self.n_steps_exact = int(n_steps_exact)
        self.n_steps_model = int(n_steps_model)
        self.n_runs = int(n_runs)
        self.name = mode_name

class FleetOperatorAgent:

    def __init__(self, NR, routes, NL, links, fleet_size, fleet_introduction, lambdaHDV, lambdaCAV, OptMode):
        self.fleet_introduction = fleet_introduction
        self.FS = fleet_size
        self.NR = NR
        self.NL = NL
        self.routes = routes
        self.links = links
        self.Intercepts = np.zeros(self.NR)
        self.lambdaHDV = lambdaHDV
        self.lambdaCAV = lambdaCAV
        self.MODE = Mode(int(OptMode[0]), int(OptMode[1]), int(OptMode[2]), OptMode[3])

    def Target(self, HDVTravelTimes, FleetTravelTimes):
        return self.lambdaHDV*HDVTravelTimes + self.lambdaCAV*FleetTravelTimes
    
    def during_optimization(self, day):
        return (self.fleet_introduction <= day)

    def make_next_routes_choice(self, day, routes, links, HDV, ND, qcount, prev_qcount=[],
                                 t_routes_previous=[], OLDHDV=None): 

        MODE = self.MODE
        if(self.during_optimization(day)):                            

            if((MODE.n_steps_exact, MODE.n_steps_model) == (1,0)):
                NewIntercepts1 = self.QuasiStationaryUpdate(qcount)
                self.Intercepts[:] = NewIntercepts1[:]

            else:
                pass  

    ###Caution: This optimization, as implemented below, works for two routes only
    def QuasiStationaryUpdate(self, qcount):
        FleetTravelTimes = np.zeros(self.FS+1)
        HDVTravelTimes = np.zeros(self.FS+1)                                   
        simulated_troutes = np.zeros((self.FS+1, self.NR))                     
        for CAVonA in range(self.FS+1):                                        
            test_intercepts = [CAVonA, self.FS-CAVonA]
            route_counts = qcount + test_intercepts
            link_counts = routesCounts2linksCounts(self.NR, self.routes, self.NL, self.links, route_counts)

            for r in range(self.NR):
                simulated_troutes[CAVonA][r] = self.routes[r].travel_time(self.links, link_counts)

            FleetTravelTimes[CAVonA] = (np.dot(simulated_troutes[CAVonA][:], test_intercepts[:]))        
            HDVTravelTimes[CAVonA] = (np.dot(simulated_troutes[CAVonA][:], qcount[:]))
        NewIntercepts = np.zeros(self.NR)
        NewIntercepts[0] = np.argmin(self.Target(HDVTravelTimes, FleetTravelTimes)[:self.FS+1]) 
        NewIntercepts[1] = self.FS-NewIntercepts[0]                                                     
        return NewIntercepts
                

############Global Functions##################

def routesCounts2linksCounts(NR, Routes, NL, Links, counts_on_routes):
    counts_on_links = np.zeros(NL)
    
    for r in range(NR):
        for l in range(len(Routes[r].link_incidency)):
            if(Routes[r].link_incidency[l] > 0):
                counts_on_links[l] += counts_on_routes[r]
    return counts_on_links

def eps_greedy(epsg, randVal, randRange, defbest):
    if randVal < epsg: return random.randint(randRange)
    else: return defbest

def BPR(t0, kappa, cap, power, count):
    return t0 + kappa * (((count) / cap) ** power)
    
def createNetwork(net_params):
    LinksNew = []
    NLNew = int(net_params["Links"]["NumberOfLinks"])
    for l in range(NLNew):
        LinksNew.append(Link(float(net_params["Links"]["t0links"][l]), float(net_params["Links"]["kappas"][l]),
                             mult*float(net_params["Links"]["Capacities"][l]), int(net_params["Links"]["Powers"][l])))
    return NLNew, LinksNew


#####The main Simulator, computes one scenario with with fixed parameters#####
def Simulator(RoadNetworks, NetworkDayChange, NL, Links, NR, Routes, Fleet_Mode = DEFAULT_FLEET_MODE, Model_Name = DEFAULT_MODEL_NAME,
              Alpha_Zero = DEFAULT_ALPHA_ZERO, Epsilon_Zero = DEFAULT_EPSILON_ZERO, Logit_Exp_Coeff = DEFAULT_LOGIT_PARAM,
              Initial_Knowledge = DEFAULT_INITIAL_KNOWLEDGE, Initial_Choice = DEFAULT_INITIAL_CHOICE, Only_Experience = DEFAULT_ONLY_EXPERIENCE,
              Fleet_size = DEFAULT_FLEET_SIZE, Fleet_introduction = DEFAULT_FLEET_INTRODUCTION,
              LambdaHDV = DEFAULT_LAMBDA_HDV, LambdaCAV = DEFAULT_LAMBDA_CAV, CAVTarget = DEFAULT_CAV_TARGET, jrange = DEFAULT_JRANGE,
              totalND = DEFAULT_TOTALND):

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

        HDV.append(HumanAgent(jrange, Model_Name, Alpha_Zero, LEARNING_RATE_DECAY, Epsilon_Zero,
                              EXPLORATION_RATE_DECAY, Qinit, INITIALCHOICE, Logit_Exp_Coeff, Only_Experience, NR))
        
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
            routeTimes[j, r] += random.normal(loc=0.0, scale=ROUTE_RANDOM_VAR)

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



###############Main##################################
#read config, network(N) and experiment (E) data from files
if __name__ == "__main__":
    params_data = params[kc.PARAMS_DATA]
    dataE = params[kc.EDATA]

    RoadNetworks = params_data[kc.ROAD_NETWORKS]
    NetworkDayChange = params_data[kc.NETWORK_DAY_CHANGE]

    net_params = params[RoadNetworks[0]]

    NL, Links = createNetwork(net_params)

    for l in range(NL):
        print("Link " + str(l))
        Links[l].print()
    Routes = []
    NR = int(net_params[kc.ROUTES][kc.NUMBER_OF_ROUTES]) 
    for r in range(NR):
        Routes.append(Route(Links, net_params[kc.ROUTES][kc.LINKS_MATRIX][r], net_params[kc.ROUTE_LABELS][r]))

    ENAME = dataE["name"]
    EXaxis = dataE["Xaxis"]
    EYaxis = dataE["Yaxis"]
    EDIRNAME = (datadirpath + ENAME + EXaxis + EYaxis)
    EXval = [(x) for x in(dataE["Xval"])]
    EYval = [(x) for x in(dataE["Yval"])]
    EXvalN = len(EXval)
    EYvalN = len(EYval) 
    EXaxisName = dataE["XaxisName"]
    EYaxisName = dataE["YaxisName"]

    try: os.mkdir(EDIRNAME)
    except: pass

    shutil.copyfile('EData.json', EDIRNAME + "/" + 'EData.json')
    shutil.copyfile(inputFileName + '.json', EDIRNAME + "/" + inputFileName+'.json')
    for rn in range(len(RoadNetworks)): shutil.copyfile(params_data[kc.ROAD_NETWORKS][rn] + '.json', EDIRNAME + "/" + params_data[kc.ROAD_NETWORKS][rn] + '.json')

    print("Experimental data loaded")
    print(str(EYaxis) + " used in experiments:" + str(EYval))
    print(str(EXaxis) + " used in experiments:" + str(EXval))

    for exind in range(EXvalN):
        for eyind in range(EYvalN):
            d = {EXaxis: EXval[exind], EYaxis: EYval[eyind]}
            print(d)                
            (TTdfs, VCdfs) = Simulator(RoadNetworks, NetworkDayChange, NL, Links, NR, Routes, **d)
            ToFiledf = TTdfs[["Day",] + ["Travel time on " + Routes[r].label for r in range(NR)] +
                                ["Mean HDV travel time",] + ["Mean HDV Perceived travel time",] + ["Mean CAV travel time",]].copy()
            ToFiledf2 = VCdfs[["Day",] + ["HDV count on " + Routes[r].label for r in range(NR)]
                                        + ["CAV count on " + Routes[r].label for r in range(NR)]].copy()
            filepath = "./" + EDIRNAME + "/" + ENAME + str('_')+ str(EXval[exind]) + str('_') + str(EYval[eyind]) + '.csv'
            ToFiledf.merge(ToFiledf2, how='left', on='Day').to_csv(filepath)
            print(EXaxisName + ": " + str(EXval[exind]) + " " + EYaxisName + ": " + str(EYval[eyind]) + " Done")    

####################

