import json
from keychain import Keychain as kc
from .link import Link

params = open(kc.INPUT_FILE)
params = json.load(params)

data_params = params[kc.PARAMS_DATA]
mult = data_params[kc.MULTIPLIER]                                              


def createNetwork(net_params):
    LinksNew = []
    NLNew = int(net_params["Links"]["NumberOfLinks"])
    for l in range(NLNew):
        LinksNew.append(Link(float(net_params["Links"]["t0links"][l]), float(net_params["Links"]["kappas"][l]),
                             mult*float(net_params["Links"]["Capacities"][l]), int(net_params["Links"]["Powers"][l])))
    return NLNew, LinksNew