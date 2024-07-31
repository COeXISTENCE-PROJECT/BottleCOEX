import json
from keychain import Keychain as kc
from .link import Link
from typing import Dict, List, Tuple                                      


def createNetwork(net_params: Dict[str, Dict[str, List[float]]], mult: float) -> Tuple[int, List['Link']]:
    """
    Create a network of links based on provided network parameters.

    Parameters:
    - net_params (Dict[str, Dict[str, List[float]]]): Nested dictionary containing network parameters.
      - "Links": Dictionary containing lists of parameters for each link.
        - "NumberOfLinks" (str): Number of links.
        - "t0links" (List[float]): List of free-flow travel times for each link.
        - "kappas" (List[float]): List of kappa values for each link.
        - "Capacities" (List[float]): List of capacities for each link.
        - "Powers" (List[float]): List of power values for each link.
    - mult (float, optional): Multiplier for capacities (default is 1.0).

    Returns:
    - Tuple[int, List[Link]]: Number of links and a list of Link objects.
    """

    LinksNew = []
    NLNew = int(net_params[kc.LINKS][kc.NUMBER_OF_LINKS])

    for l in range(NLNew):

        LinksNew.append(Link(float(net_params[kc.LINKS][kc.T0LINKS][l]), float(net_params[kc.LINKS][kc.KAPPAS][l]),
                             mult*float(net_params[kc.LINKS][kc.CAPACITIES][l]), int(net_params[kc.LINKS][kc.POWERS][l])))
        
    return NLNew, LinksNew