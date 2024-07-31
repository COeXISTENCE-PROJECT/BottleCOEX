import numpy as np
import random
from typing import List, Dict, Tuple
from .utils import eps_greedy
from keychain import Keychain as kc

class HumanAgent:
    def __init__(self, human_params: Dict[str, float], jrange: int, model_name: str, learning_rate: float, 
                 learning_rate_decay: float, exploration_rate: float, exploration_rate_decay: float,
                 Qinit: List[float], Qinitchoice: str, logit_par: float, only_experience: bool, NR: int):
        """
        Initialize a HumanAgent object.

        Parameters:
        - human_params (Dict[str, float]): Dictionary of human parameters.
        - jrange (int): Range for storing Q values.
        - model_name (str): Name of the model.
        - learning_rate (float): Learning rate for Q-value updates.
        - learning_rate_decay (float): Decay rate for the learning rate.
        - exploration_rate (float): Exploration rate for route choice.
        - exploration_rate_decay (float): Decay rate for the exploration rate.
        - Qinit (List[float]): Initial Q values.
        - Qinitchoice (str): Initial route choice strategy ('Argmin' or 'Random').
        - logit_par (float): Logit parameter.
        - only_experience (bool): Flag to learn only from experience.
        - NR (int): Number of routes.
        """
        self.NR = NR
        self.lr = learning_rate
        self.er = exploration_rate
        self.initlr = self.lr
        self.initer = self.er
        self.lr_decay = learning_rate_decay
        self.er_decay = exploration_rate_decay
        self.Q = np.array(Qinit[:])
        self.Qstored = np.zeros((jrange, self.NR))
        
        if Qinitchoice == 'Argmin':
            self.curRoute = np.argmin(self.Q[:])
        elif Qinitchoice == 'Random':
            self.curRoute = random.randint(0, self.NR - 1)
        else:
            raise ValueError("Initial choice not specified")
        
        self.minRoute = 0
        self.learning_ctr = 0
        self.human_params = human_params
        self.learns = self.human_params[kc.LEARNING]
        self.learn_from_experience = only_experience
        self.logit_par = logit_par
        gumbelScale = 1 / self.logit_par
        gumbelMean = 0
        gumbelLocation = gumbelMean - (gumbelScale * np.euler_gamma)  # Euler-Mascheroni constant
        self.gumbelOffsets = np.random.gumbel(gumbelLocation, gumbelScale, self.NR)
        self.model_name = model_name
        if self.model_name == "Gumbel":
            self.er = 0

    def learn(self, t_routes: np.ndarray) -> None:
        """
        Update Q values based on travel times.

        Parameters:
        - t_routes (np.ndarray): Array of travel times for each route.
        """
        self.Qstored[self.learning_ctr, :] = self.Q[:]
        self.learning_ctr += 1
        if self.learns:
            if self.learn_from_experience:
                noise = np.random.normal(loc=0.0, scale=self.human_params[kc.DRIVER_RANDOM_VAR])
                self.Q[self.curRoute] = (1 - self.lr) * self.Q[self.curRoute] + self.lr * (t_routes[self.curRoute] + noise)
            else:
                self.Q[:] = (1 - self.lr) * self.Q[:] + self.lr * t_routes[:]
            if self.Q[self.curRoute] < self.Q[self.minRoute]:
                self.minRoute = self.curRoute

    def update_rates(self) -> None:
        """
        Update the exploration and learning rates.
        """
        self.er *= self.er_decay
        self.lr *= self.lr_decay

    def make_next_route_choice(self) -> None:
        """
        Make the next route choice based on the current model.
        """
        if self.model_name == "Gumbel":
            self.make_Gumbel_choice(random.random())
        elif self.model_name == "GumbelEps":
            self.make_Gumbel_choice(random.random())
        elif self.model_name == "Logit":
            self.make_Logit_choice(random.random())
        elif self.model_name == "EpsGreedy":
            self.make_epsilon_greedy_choice(random.random())
        else:
            raise ValueError("Choice model not specified")

    def make_epsilon_greedy_choice(self, random_value: float) -> None:
        """
        Make a route choice using epsilon-greedy strategy.

        Parameters:
        - random_value (float): Random value for exploration decision.
        """
        minindexEG = np.random.choice(np.flatnonzero(self.Q[:] == self.Q[:].min()))
        if random_value < self.er:
            self.curRoute = random.randint(0, self.NR - 1)
        else:
            self.curRoute = minindexEG

    def make_Gumbel_choice(self, random_value: float) -> None:
        """
        Make a route choice using Gumbel distribution.

        Parameters:
        - random_value (float): Random value for exploration decision.
        """
        QG = self.Q[:] + self.gumbelOffsets[:]
        self.curRoute = eps_greedy(self.er, random_value, self.NR, np.argmin(QG[:]))

    def make_Logit_choice(self, random_value: float) -> None:
        """
        Make a route choice using logit model.

        Parameters:
        - random_value (float): Random value for exploration decision.
        """
        exps = np.exp(-self.logit_par * self.Q[:])
        denominator = np.sum(exps)
        epsilons = exps / denominator
        cumulative_prob = 0.0

        for r in range(self.NR):
            cumulative_prob += epsilons[r]
            if random_value < cumulative_prob:
                self.curRoute = r
                break
