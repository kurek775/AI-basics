import numpy as np


class SARSA_Learning:

    def __init__(self, env, alpha, gamma, epsilon, numberEpisodes):

        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.stateNumber = env.observation_space.n
        self.actionNumber = env.action_space.n
        self.numberEpisodes = numberEpisodes
        self.learnedPolicy = np.zeros(env.observation_space.n)
        self.Qmatrix = np.zeros((self.stateNumber, self.actionNumber))


    def selectAction(self, state, index):

        if index < 100:
            return np.random.choice(self.actionNumber)
 
        randomNumber = np.random.random()

        if index > 1000:
            self.epsilon = 0.9*self.epsilon

    
        if randomNumber < self.epsilon:
            return np.random.choice(self.actionNumber)
        else:
            return np.random.choice(np.where(self.Qmatrix[state, :] == np.max(self.Qmatrix[state, :]))[0])

    def simulateEpisodes(self):
        for indexEpisode in range(self.numberEpisodes):
            (stateS, prob) = self.env.reset()
            actionA = self.selectAction(stateS, indexEpisode)

            print("Simulating episode {}".format(indexEpisode))
            terminalState = False
            while not terminalState:
                (stateSprime, rewardPrime, terminalState,
                 _, _) = self.env.step(actionA)
                
                actionAprime = self.selectAction(stateSprime, indexEpisode)

                if not terminalState:
                    error = rewardPrime+self.gamma * \
                        self.Qmatrix[stateSprime, actionAprime] - \
                        self.Qmatrix[stateS, actionA]
                    self.Qmatrix[stateS, actionA] = self.Qmatrix[stateS,
                                                                 actionA]+self.alpha*error
                else:
                    error = rewardPrime-self.Qmatrix[stateS, actionA]
                    self.Qmatrix[stateS, actionA] = self.Qmatrix[stateS,
                                                                 actionA]+self.alpha*error
                
                print("State: {}, Action: {}, Reward: {}, Next State: {}, Next Action: {}".format(
                    stateS, actionA, rewardPrime, stateSprime, actionAprime))

                stateS = stateSprime
                actionA = actionAprime

    def computeFinalPolicy(self):
        for indexS in range(self.stateNumber):
            self.learnedPolicy[indexS] = np.random.choice(
                np.where(self.Qmatrix[indexS] == np.max(self.Qmatrix[indexS]))[0])
