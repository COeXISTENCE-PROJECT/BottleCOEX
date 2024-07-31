from keychain import Keychain as kc
import numpy as np
import random
from .utils import eps_greedy

class HumanAgent:

    def __init__(self, human_params, jrange, model_name, learning_rate, learning_rate_decay, exploration_rate, exploration_rate_decay,
                 Qinit, Qinitchoice, logit_par, only_experience, NR):
        self.NR = NR
        self.lr = learning_rate
        self.er = exploration_rate
        self.initlr = self.lr
        self.initer = self.er
        self.lr_decay = learning_rate_decay
        self.er_decay = exploration_rate_decay
        self.Q = np.zeros(self.NR)
        self.Q[:] = Qinit[:]
        self.Qstored = np.zeros((jrange, self.NR))
        if(Qinitchoice == 'Argmin'):  self.curRoute = np.argmin(self.Q[:])
        elif(Qinitchoice == 'Random'):self.curRoute = random.randint(0, self.NR - 1)
        else: raise Exception("Initial choice not specified")
        self.minRoute = 0
        self.learning_ctr = 0
        self.human_params = human_params
        self.learns = self.human_params[kc.LEARNING]
        self.learn_from_experience = only_experience
        self.logit_par = logit_par
        gumbelScale = 1/self.logit_par  
        gumbelMean = 0
        gumbelLocation = gumbelMean - (gumbelScale*np.euler_gamma)            #location of Gumbel can be computed using the scale and mean only (and Euler-Mascheroni constant)
        self.gumbelOffsets = np.random.gumbel(gumbelLocation, gumbelScale, self.NR)        
        self.model_name = model_name
        if(self.model_name == "Gumbel"): self.er = 0

    def learn(self, t_routes):
        self.Qstored[self.learning_ctr,:] = self.Q[:]
        self.learning_ctr += 1
        if(self.learns):
            if(self.learn_from_experience):
                self.Q[self.curRoute] = (1-self.lr) * self.Q[self.curRoute] + (self.lr*(t_routes[self.curRoute]
                                                                                     + np.random.normal(loc=0.0, scale=self.human_params[kc.DRIVER_RANDOM_VAR])))
            else:
                self.Q[:] = (1-self.lr)*self.Q[:] + self.lr*(t_routes[:]) 
            if self.Q[self.curRoute] < self.Q[self.minRoute]:
                self.minRoute = self.curRoute

    def update_rates(self):
        self.er *= self.er_decay
        self.lr *= self.lr_decay

    def make_next_route_choice(self):
        if(self.model_name == "Gumbel"):
            self.make_Gumbel_choice(random.random())
        elif(self.model_name == "GumbelEps"):
            self.make_Gumbel_choice(random.random())
        elif(self.model_name == "Logit"):
            self.make_Logit_choice(random.random())                                   
        elif(self.model_name == "EpsGreedy"):
            self.make_epsilon_greedy_choice(random.random())
        else: raise Exception("Choice model not specified")
        
    def make_epsilon_greedy_choice(self, random_value):
        minindexEG = np.random.choice(np.flatnonzero(self.Q[:] == self.Q[:].min()))                                         
        if (random_value < self.er): self.curRoute = random.randint(self.NR)
        else: self.curRoute = minindexEG
        
    def make_Gumbel_choice(self, random_value):
        QG = self.Q[:] + self.gumbelOffsets[:]                                          
        self.curRoute = eps_greedy(self.er, random_value, self.NR, np.argmin(QG[:]))         

    def make_Logit_choice(self, random_value):
        exps = np.exp(-self.logit_par*self.Q[:])
        denominator = np.sum(exps)
        epsilons = exps / denominator
        r=0
        threshold = 0.0
        while(r < self.NR):
            threshold += epsilons[r]
            if(random_value < threshold):
                self.curRoute = r
                r = self.NR
            r = r+1                
