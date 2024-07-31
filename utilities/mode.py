
class Mode:
    
    def __init__(self, n_steps_exact: int, n_steps_model: int, n_runs: int = 1, mode_name: str = ''):
        """
        Initialize a Mode object.

        Parameters:
        - n_steps_exact (int): Number of exact steps.
        - n_steps_model (int): Number of model steps.
        - n_runs (int, optional): Number of runs (default is 1).
        - mode_name (str, optional): Name of the mode (default is an empty string).
        """
        self.n_steps_exact = n_steps_exact
        self.n_steps_model = n_steps_model
        self.n_runs = n_runs
        self.name = mode_name