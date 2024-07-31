import numpy as np
from typing import List

class Route:

    def __init__(self, links: List, link_incidency: List[int], label: str):
        """
        Initialize a Route object.

        Parameters:
        - links (List[Link]): List of Link objects.
        - link_incidency (List[int]): List indicating the incidence of each link in the route.
        - label (str): Label or name of the route.
        """
        self.link_incidency = link_incidency
        self.label = label
        self.t0 = self.calculate_travel_time(links, np.zeros(len(links)))
        
    def calculate_travel_time(self, links: List, counts_on_links: np.ndarray) -> float:
        """
        Calculate the total travel time for the route based on the traffic counts on each link.

        Parameters:
        - counts_on_links (np.ndarray): Array of traffic counts on each link.

        Returns:
        - float: Total travel time for the route.
        """
        total_travel_time  = 0

        for l in range(len(self.link_incidency)):
            if(self.link_incidency[l] > 0):
                total_travel_time  = total_travel_time  + links[l].travel_time(counts_on_links[l])

        return total_travel_time 