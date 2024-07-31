import numpy as np


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