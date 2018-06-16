# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discountRate = 0.9, iters = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.

      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discountRate = discountRate
    self.iters = iters
    self.values = util.Counter() # A Counter is a dict with default 0

    """Description:
    [I updated all the values of the states through all the given number of 
     iterations.]
    """
    """ YOUR CODE HERE """
    states = self.mdp.getStates()
    
    for i in range(iters-1):
        currentValues = self.values.copy()
        for state in states:
            currentValues[state] = self.getValue(state)
        self.values = currentValues
            
    """ END CODE """

  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """

    """Description:
    [This functions returns the value of the given state. It returns the max of
     all QValues for every legal action from the state.]
    """
    """ YOUR CODE HERE """
    if self.mdp.isTerminal(state):
        return 0
    actions = self.mdp.getPossibleActions(state)
    value = None
    for act in actions:
        temp = self.getQValue(state,act)
        if value==None or temp>value:
            value = temp
    if value==None:
        value = 0
    return value
    """ END CODE """

  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    """Description:
    [Returns the QValue for the (state,action) pair with the equation: Q(s,a) =
     Sum(s')(T(s,a,s')*[R(s,a,s) + self.discountRate*V*(s')])]
    """
    """ YOUR CODE HERE """
    probs = self.mdp.getTransitionStatesAndProbs(state,action)
    value = 0
    for nextState,probability in probs:
        reward = self.mdp.getReward(state,action,nextState)
        discount = self.values[nextState] * self.discountRate
        total = probability * (reward + discount)
        value += total
    return value  
    """ END CODE """

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """

    """Description:
    [Returns the action from the state that gives the highest Qvalue.]
    """
    """ YOUR CODE HERE """
    if self.mdp.isTerminal(state):
      return None
    move = None
    maximum = None
    for action in self.mdp.getPossibleActions(state):
        temp = self.getQValue(state,action)
        if maximum==None or temp>maximum + 0.000001:
            maximum = temp
            move = action
    return move
    #return self.policies[state]
    """ END CODE """

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
