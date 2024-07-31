from .utils import BPR

class Link:
    def __init__(self, t0: float, kappa: float, cap: float, power: float):
        """
        Initialize a Link object.

        Parameters:
        - t0 (float): Free-flow travel time.
        - kappa (float): Parameter that affects travel time.
        - cap (float): Capacity of the link.
        - power (float): Power parameter for the BPR function.
        """
        self.t0 = t0
        self.kappa = kappa
        self.cap = cap
        self.power = power

    def print(self) -> str:
        """
        Return a string representation of the Link object.
        """
        return (f"t0: {self.t0}, cap: {self.cap}, "
                f"kappa: {self.kappa}, power: {self.power}")

    def travel_time(self, count: float) -> float:
        """
        Calculate the travel time based on the BPR function.

        Parameters:
        - count (float): Traffic count or flow.

        Returns:
        - float: Computed travel time.
        """
        return BPR(self.t0, self.kappa, self.cap, self.power, count)
