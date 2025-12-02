# ghostAgents.py
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


from game import Agent
from game import Actions
from game import Directions
import random
from util import manhattanDistance
import util


class GhostAgent(Agent):
    def __init__(self, index):
        self.index = index

    def getAction(self, state):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution(dist)

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()


class RandomGhost(GhostAgent):
    "A ghost that chooses a legal action uniformly at random."

    def getDistribution(self, state):
        dist = util.Counter()
        for a in state.getLegalActions(self.index):
            dist[a] = 1.0
        dist.normalize()
        return dist


class DirectionalGhost(GhostAgent):
    "A ghost that prefers to rush Pacman, or flee when scared."

    def __init__(self, index, prob_attack=0.8, prob_scaredFlee=0.8):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution(self, state):
        # Read variables from state
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pos = state.getGhostPosition(self.index)
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared:
            speed = 0.5

        actionVectors = [Actions.directionToVector(
            a, speed) for a in legalActions]
        newPositions = [(pos[0]+a[0], pos[1]+a[1]) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance(
            pos, pacmanPosition) for pos in newPositions]
        if isScared:
            bestScore = max(distancesToPacman)
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min(distancesToPacman)
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip(
            legalActions, distancesToPacman) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions:
            dist[a] = bestProb / len(bestActions)
        for a in legalActions:
            dist[a] += (1-bestProb) / len(legalActions)
        dist.normalize()
        return dist


class AmbushGhost(GhostAgent):
    """
    A ghost that targets a position AHEAD of Pac-Man (like Pinky in original game).
    This creates ambush behavior - the ghost tries to cut off Pac-Man's path.
    """

    def __init__(self, index, prob_ambush=0.75, look_ahead=4):
        self.index = index
        self.prob_ambush = prob_ambush
        self.look_ahead = look_ahead  # How many tiles ahead to target

    def getDistribution(self, state):
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pos = state.getGhostPosition(self.index)
        isScared = ghostState.scaredTimer > 0

        speed = 0.5 if isScared else 1

        actionVectors = [Actions.directionToVector(a, speed) for a in legalActions]
        newPositions = [(pos[0]+a[0], pos[1]+a[1]) for a in actionVectors]
        
        pacmanPosition = state.getPacmanPosition()
        pacmanDirection = state.getPacmanState().getDirection()
        
        # Calculate target position (ahead of Pac-Man)
        if isScared:
            # When scared, run away from Pac-Man
            targetPos = pacmanPosition
            distancesToTarget = [manhattanDistance(p, targetPos) for p in newPositions]
            bestScore = max(distancesToTarget)
            bestProb = self.prob_ambush
        else:
            # Target position ahead of Pac-Man based on his direction
            dx, dy = Actions.directionToVector(pacmanDirection, self.look_ahead)
            targetPos = (pacmanPosition[0] + dx, pacmanPosition[1] + dy)
            
            # If target is too far or in a wall, just chase directly
            distToTarget = manhattanDistance(pos, targetPos)
            distToPacman = manhattanDistance(pos, pacmanPosition)
            
            if distToTarget > distToPacman + 4:
                targetPos = pacmanPosition
            
            distancesToTarget = [manhattanDistance(p, targetPos) for p in newPositions]
            bestScore = min(distancesToTarget)
            bestProb = self.prob_ambush

        bestActions = [action for action, distance in zip(legalActions, distancesToTarget) 
                       if distance == bestScore]

        dist = util.Counter()
        for a in bestActions:
            dist[a] = bestProb / len(bestActions)
        for a in legalActions:
            dist[a] += (1-bestProb) / len(legalActions)
        dist.normalize()
        return dist


class FlankingGhost(GhostAgent):
    """
    A ghost that coordinates with another ghost to flank Pac-Man (like Inky).
    Uses the position of ghost index 1 (Blinky) as reference point.
    """

    def __init__(self, index, prob_flank=0.7, reference_ghost=1):
        self.index = index
        self.prob_flank = prob_flank
        self.reference_ghost = reference_ghost

    def getDistribution(self, state):
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pos = state.getGhostPosition(self.index)
        isScared = ghostState.scaredTimer > 0

        speed = 0.5 if isScared else 1

        actionVectors = [Actions.directionToVector(a, speed) for a in legalActions]
        newPositions = [(pos[0]+a[0], pos[1]+a[1]) for a in actionVectors]
        
        pacmanPosition = state.getPacmanPosition()
        
        if isScared:
            # When scared, run away
            distancesToPacman = [manhattanDistance(p, pacmanPosition) for p in newPositions]
            bestScore = max(distancesToPacman)
            bestProb = self.prob_flank
            bestActions = [action for action, distance in zip(legalActions, distancesToPacman) 
                           if distance == bestScore]
        else:
            # Calculate flanking position based on reference ghost
            try:
                refPos = state.getGhostPosition(self.reference_ghost)
                # Target is on opposite side of Pac-Man from reference ghost
                # Vector from reference ghost to Pac-Man, extended past Pac-Man
                vec_x = pacmanPosition[0] - refPos[0]
                vec_y = pacmanPosition[1] - refPos[1]
                targetPos = (pacmanPosition[0] + vec_x, pacmanPosition[1] + vec_y)
            except:
                # Fallback to direct chase
                targetPos = pacmanPosition
            
            distancesToTarget = [manhattanDistance(p, targetPos) for p in newPositions]
            bestScore = min(distancesToTarget)
            bestProb = self.prob_flank
            bestActions = [action for action, distance in zip(legalActions, distancesToTarget) 
                           if distance == bestScore]

        dist = util.Counter()
        for a in bestActions:
            dist[a] = bestProb / len(bestActions)
        for a in legalActions:
            dist[a] += (1-bestProb) / len(legalActions)
        dist.normalize()
        return dist


class PatrolGhost(GhostAgent):
    """
    A ghost that patrols near Pac-Man but keeps some distance (like Clyde).
    Chases when far, but backs off when too close.
    """

    def __init__(self, index, prob_patrol=0.7, safe_distance=8):
        self.index = index
        self.prob_patrol = prob_patrol
        self.safe_distance = safe_distance

    def getDistribution(self, state):
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pos = state.getGhostPosition(self.index)
        isScared = ghostState.scaredTimer > 0

        speed = 0.5 if isScared else 1

        actionVectors = [Actions.directionToVector(a, speed) for a in legalActions]
        newPositions = [(pos[0]+a[0], pos[1]+a[1]) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()
        
        currentDist = manhattanDistance(pos, pacmanPosition)
        distancesToPacman = [manhattanDistance(p, pacmanPosition) for p in newPositions]

        if isScared:
            # When scared, always run away
            bestScore = max(distancesToPacman)
            bestProb = self.prob_patrol
        elif currentDist > self.safe_distance:
            # Far away: chase Pac-Man
            bestScore = min(distancesToPacman)
            bestProb = self.prob_patrol
        else:
            # Too close: back off to patrol distance
            bestScore = max(distancesToPacman)
            bestProb = self.prob_patrol * 0.6  # Less committed when backing off

        bestActions = [action for action, distance in zip(legalActions, distancesToPacman) 
                       if distance == bestScore]

        dist = util.Counter()
        for a in bestActions:
            dist[a] = bestProb / len(bestActions)
        for a in legalActions:
            dist[a] += (1-bestProb) / len(legalActions)
        dist.normalize()
        return dist


class AggressiveGhost(GhostAgent):
    """
    A highly aggressive ghost that always chases (95% probability).
    Used for harder difficulty training.
    """

    def __init__(self, index, prob_chase=0.95):
        self.index = index
        self.prob_chase = prob_chase

    def getDistribution(self, state):
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pos = state.getGhostPosition(self.index)
        isScared = ghostState.scaredTimer > 0

        speed = 0.5 if isScared else 1

        actionVectors = [Actions.directionToVector(a, speed) for a in legalActions]
        newPositions = [(pos[0]+a[0], pos[1]+a[1]) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        distancesToPacman = [manhattanDistance(p, pacmanPosition) for p in newPositions]
        
        if isScared:
            bestScore = max(distancesToPacman)
        else:
            bestScore = min(distancesToPacman)
        
        bestProb = self.prob_chase
        bestActions = [action for action, distance in zip(legalActions, distancesToPacman) 
                       if distance == bestScore]

        dist = util.Counter()
        for a in bestActions:
            dist[a] = bestProb / len(bestActions)
        for a in legalActions:
            dist[a] += (1-bestProb) / len(legalActions)
        dist.normalize()
        return dist
