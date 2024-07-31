import numpy as np
import random


def routesCounts2linksCounts(NR, Routes, NL, Links, counts_on_routes):
    counts_on_links = np.zeros(NL)
    
    for r in range(NR):
        for l in range(len(Routes[r].link_incidency)):
            if(Routes[r].link_incidency[l] > 0):
                counts_on_links[l] += counts_on_routes[r]
    return counts_on_links

def eps_greedy(epsg, randVal, randRange, defbest):
    if randVal < epsg: return random.randint(0, randRange - 1)
    else: return defbest

def BPR(t0, kappa, cap, power, count):
    return t0 + kappa * (((count) / cap) ** power)
    