from .utils import BPR

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