# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        pacmanPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        foodList = newFood.asList()
        if foodList:
            score += sum(1.0 / manhattanDistance(pacmanPos, food) for food in foodList)

        ghostDistances = [manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in newGhostStates]
        for i, ghostDistance in enumerate(ghostDistances):
            if newScaredTimes[i] == 0 and ghostDistance > 0:
                score -= 1.0 / ghostDistance


        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)

    """
    def minimax(self, gameState, depth, agentIndex):
        if (gameState.isWin() or gameState.isLose() or depth == self.depth):
            return self.evaluationFunction(gameState)
        if agentIndex == 0:  # max
            bestValue = float('-inf')

            for action in gameState.getLegalActions(agentIndex):

                successor = gameState.generateSuccessor(agentIndex, action)
                value = self.minimax(successor, depth, 1)  # Next agent (ghost)
                bestValue = max(bestValue, value)

            return bestValue
        else:  # min
            bestValue = float('inf')
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = depth + 1 if nextAgentIndex == 0 else depth

            for action in gameState.getLegalActions(agentIndex):

                successor = gameState.generateSuccessor(agentIndex, action)
                value = self.minimax(successor, nextDepth, nextAgentIndex)
                bestValue = min(bestValue, value)

            return bestValue
        

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        bestScore = float('-inf')
        bestAction = None

        for action in gameState.getLegalActions(0): 
            successor = gameState.generateSuccessor(0, action)
            score = self.minimax(successor, 0, 1)  
            if score > bestScore:
                bestScore = score
                bestAction = action 

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState, alpha, beta)

            else:
                return minValue(agentIndex, depth, gameState, alpha, beta)

        def maxValue(agentIndex, depth, gameState, alpha, beta):
            legalActions = gameState.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(gameState)

            bestScore = float('-inf')

            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                bestScore = max(bestScore, alphaBeta(1, depth, successorState, alpha, beta))

                alpha = max(alpha, bestScore)

                if alpha > beta: 
                    break

            return bestScore

        def minValue(agentIndex, depth, gameState, alpha, beta):
            legalActions = gameState.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(gameState)

            bestScore = float('inf')

            numAgents = gameState.getNumAgents()

            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)

                if agentIndex == numAgents - 1:
                    bestScore = min(bestScore, alphaBeta(0, depth + 1, successorState, alpha, beta))
                else:
                    bestScore = min(bestScore, alphaBeta(agentIndex + 1, depth, successorState, alpha, beta))

                beta = min(beta, bestScore)

                if alpha > beta: 
                    break

            return bestScore

        alpha = float('-inf')
        beta = float('inf')

        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestScore = float('-inf')

        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            score = alphaBeta(1, 0, successorState, alpha, beta)

            if score > bestScore:
                bestScore = score
                bestAction = action

            alpha = max(alpha, bestScore)

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState)

            else:
                return expValue(agentIndex, depth, gameState)

        def maxValue(agentIndex, depth, gameState):
            legalActions = gameState.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(gameState)

            bestScore = float('-inf')

            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                bestScore = max(bestScore, expectimax(1, depth, successorState))

            return bestScore

        def expValue(agentIndex, depth, gameState):
            legalActions = gameState.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(gameState)

            totalScore = 0
            numActions = len(legalActions)

            numAgents = gameState.getNumAgents()

            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)

                if agentIndex == numAgents - 1:
                    totalScore += expectimax(0, depth + 1, successorState)
                else:
                    totalScore += expectimax(agentIndex + 1, depth, successorState)

            return totalScore / numActions

        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestScore = float('-inf')

        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            score = expectimax(1, 0, successorState)

            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')

    score = currentGameState.getScore()
    pacmanPos = currentGameState.getPacmanPosition()

    foodList = currentGameState.getFood().asList()


    if foodList:
        score += sum(2.0 / manhattanDistance(pacmanPos, food) for food in foodList)

    capsules = currentGameState.getCapsules()
    score -= 20 * len(capsules)  

    for ghost in currentGameState.getGhostStates():
        dist = manhattanDistance(pacmanPos, ghost.getPosition())

        if ghost.scaredTimer > 0:
            score += 200.0 / (dist + 1)
        else:
            if dist <= 1:
                score -= 400
            else:
                score -= 10.0 / dist

    return score

# Abbreviation
better = betterEvaluationFunction
