import numpy as np
from .mode import Mode
from .utils import routesCounts2linksCounts

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
                simulated_troutes[CAVonA][r] = self.routes[r].calculate_travel_time(self.links, link_counts)

            FleetTravelTimes[CAVonA] = (np.dot(simulated_troutes[CAVonA][:], test_intercepts[:]))        
            HDVTravelTimes[CAVonA] = (np.dot(simulated_troutes[CAVonA][:], qcount[:]))
        NewIntercepts = np.zeros(self.NR)
        NewIntercepts[0] = np.argmin(self.Target(HDVTravelTimes, FleetTravelTimes)[:self.FS+1]) 
        NewIntercepts[1] = self.FS-NewIntercepts[0]                                                     
        return NewIntercepts