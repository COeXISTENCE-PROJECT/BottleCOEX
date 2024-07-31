import numpy as np
from typing import List, Tuple, Optional

from .mode import Mode
from .utils import routesCounts2linksCounts



class FleetOperatorAgent:
    def __init__(self, NR: int, routes: List, NL: int, links: List, fleet_size: int, fleet_introduction: int,
                 lambdaHDV: float, lambdaCAV: float, OptMode: Tuple[int, int, int, str]):
        """
        Initialize a FleetOperatorAgent object.

        Parameters:
        - NR (int): Number of routes.
        - routes (List[Route]): List of Route objects.
        - NL (int): Number of links.
        - links (List[Link]): List of Link objects.
        - fleet_size (int): Size of the fleet.
        - fleet_introduction (int): Day of fleet introduction.
        - lambdaHDV (float): Lambda value for HDV.
        - lambdaCAV (float): Lambda value for CAV.
        - OptMode (Tuple[int, int, int, str]): Optimization mode parameters.
        """
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

    def Target(self, HDVTravelTimes: np.ndarray, FleetTravelTimes: np.ndarray) -> np.ndarray:
        """
        Calculate the target function value.

        Parameters:
        - HDVTravelTimes (np.ndarray): Array of HDV travel times.
        - FleetTravelTimes (np.ndarray): Array of Fleet travel times.

        Returns:
        - np.ndarray: Calculated target values.
        """
        return self.lambdaHDV * HDVTravelTimes + self.lambdaCAV * FleetTravelTimes
    
    def during_optimization(self, day: int) -> bool:
        """
        Check if the optimization is ongoing based on the current day.

        Parameters:
        - day (int): Current day.

        Returns:
        - bool: True if the fleet introduction day is less than or equal to the current day, otherwise False.
        """
        return self.fleet_introduction <= day

    def make_next_routes_choice(self, day: int, routes: List, links: List, HDV: float, ND: float, qcount: np.ndarray,
                                prev_qcount: Optional[np.ndarray] = [], t_routes_previous: Optional[np.ndarray] = [], OLDHDV: Optional[float] = None) -> None:
        """
        Make the next route choice based on the current day and other parameters.

        Parameters:
        - day (int): Current day.
        - routes (List[Route]): List of Route objects.
        - links (List[Link]): List of Link objects.
        - HDV (float): HDV parameter.
        - ND (float): ND parameter.
        - qcount (np.ndarray): Array of counts.
        - prev_qcount (Optional[np.ndarray]): Previous counts (default is empty list).
        - t_routes_previous (Optional[np.ndarray]): Previous travel times (default is empty list).
        - OLDHDV (Optional[float]): OLDHDV parameter (default is None).
        """
        MODE = self.MODE
        if self.during_optimization(day):
            if (MODE.n_steps_exact, MODE.n_steps_model) == (1, 0):
                new_intercepts = self.QuasiStationaryUpdate(qcount)
                self.Intercepts[:] = new_intercepts[:]
            else:
                pass

    def QuasiStationaryUpdate(self, qcount: np.ndarray) -> np.ndarray:
        """
        Perform a quasi-stationary update of intercepts.

        Parameters:
        - qcount (np.ndarray): Array of counts.

        Returns:
        - np.ndarray: Updated intercepts.
        """
        fleet_travel_times = np.zeros(self.FS + 1)
        hdv_travel_times = np.zeros(self.FS + 1)
        simulated_troutes = np.zeros((self.FS + 1, self.NR))

        for cav_on_a in range(self.FS + 1):
            test_intercepts = [cav_on_a, self.FS - cav_on_a]
            route_counts = qcount + test_intercepts
            link_counts = routesCounts2linksCounts(self.NR, self.routes, self.NL, self.links, route_counts)

            for r in range(self.NR):
                simulated_troutes[cav_on_a][r] = self.routes[r].calculate_travel_time(self.links, link_counts)

            fleet_travel_times[cav_on_a] = np.dot(simulated_troutes[cav_on_a][:], test_intercepts[:])
            hdv_travel_times[cav_on_a] = np.dot(simulated_troutes[cav_on_a][:], qcount[:])

        new_intercepts = np.zeros(self.NR)
        new_intercepts[0] = np.argmin(self.Target(hdv_travel_times, fleet_travel_times)[:self.FS + 1])
        new_intercepts[1] = self.FS - new_intercepts[0]

        return new_intercepts
