import numpy as np
import random
from typing import List

def routesCounts2linksCounts(NR: int, Routes: List, NL: int, Links: List, counts_on_routes: np.ndarray) -> np.ndarray:
    """
    Convert route counts to link counts.

    Parameters:
    - NR (int): Number of routes.
    - Routes (List[Route]): List of Route objects.
    - NL (int): Number of links.
    - Links (List[Link]): List of Link objects.
    - counts_on_routes (np.ndarray): Array of counts on routes.

    Returns:
    - np.ndarray: Array of counts on links.
    """
    counts_on_links = np.zeros(NL)
    
    for r in range(NR):
        for l in range(len(Routes[r].link_incidency)):
            if Routes[r].link_incidency[l] > 0:
                counts_on_links[l] += counts_on_routes[r]
                
    return counts_on_links

def eps_greedy(epsg: float, randVal: float, randRange: int, defbest: int) -> int:
    """
    Epsilon-greedy selection strategy.

    Parameters:
    - epsg (float): Epsilon value for exploration probability.
    - randVal (float): Random value for comparison.
    - randRange (int): Range for random selection.
    - defbest (int): Default best selection.

    Returns:
    - int: Selected index.
    """
    if randVal < epsg:
        return random.randint(0, randRange - 1)
    else:
        return defbest

def BPR(t0: float, kappa: float, cap: float, power: float, count: float) -> float:
    """
    Bureau of Public Roads (BPR) function to calculate travel time.

    Parameters:
    - t0 (float): Free-flow travel time.
    - kappa (float): Parameter that affects travel time.
    - cap (float): Capacity of the link.
    - power (float): Power parameter for the BPR function.
    - count (float): Traffic count.

    Returns:
    - float: Computed travel time.
    """
    return t0 + kappa * ((count / cap) ** power)
