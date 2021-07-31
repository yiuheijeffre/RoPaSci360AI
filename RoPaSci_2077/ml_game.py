import sys
import time
import logging
import collections
import numpy as np
from random import choice, randint
import random


from ml_player import Player

np.random.seed(2)

# all hexes
_HEX_RANGE = range(-4, +4 + 1)
_ORD_HEXES = [
    (r, q) for r in _HEX_RANGE for q in _HEX_RANGE if -r - q in _HEX_RANGE
]
_SET_HEXES = frozenset(_ORD_HEXES)

# nearby hexes
_HEX_STEPS = [(1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1)]


def _ADJACENT(x):
    rx, qx = x
    return _SET_HEXES & {(rx + ry, qx + qy) for ry, qy in _HEX_STEPS}


# rock-paper-scissors mechanic
_BEATS_WHAT = {"r": "s", "p": "r", "s": "p"}
_WHAT_BEATS = {"r": "p", "p": "s", "s": "r"}


def _BATTLE(symbols):
    types = {s.lower() for s in symbols}
    if len(types) == 1:
        # no fights
        return symbols
    if len(types) == 3:
        # everyone dies
        return []
    # else there are two, only some die:
    for t in types:
        # those who are not defeated stay
        symbols = [s for s in symbols if s.lower() != _BEATS_WHAT[t]]
    return symbols

# draw conditions
_MAX_TURNS = 360  # per player

class Game:
    """
    Represent the evolving state of a game. Main useful methods
    are __init__, update, over, end, and __str__.
    """

    def __init__(self, log_filename=None):
        # initialise game board state, and both players with zero throws
        self.board = {x: [] for x in _ORD_HEXES}
        self.throws = {"upper": 0, "lower": 0}

        # also keep track of some other state variables for win/draw
        # detection (number of turns, state history)
        self.nturns = 0
        self.history = collections.Counter({self._snap(): 1})
        self.result = None

        if log_filename is not None:
            self.logger = logging.getLogger(name=log_filename)
            handler = logging.FileHandler(log_filename, mode="w")
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logging.getLogger()  # logger with no handlers

    def update(self, upper_action, lower_action):
        """
        Submit an action to the game for validation and application.
        If the action is not allowed, raise an InvalidActionException with
        a message describing allowed actions.
        Otherwise, apply the action to the game state.
        """
        # validate the actions:
        for action, c in [(upper_action, "upper"), (lower_action, "lower")]:
            actions = list(self._available_actions(c))
        # otherwise, apply the actions:
        battles = []
        atype, *aargs = upper_action
        if atype == "THROW":
            s, x = aargs
            self.board[x].append(s.upper())
            self.throws["upper"] += 1
            battles.append(x)
        else:
            x, y = aargs
            # remove ONE UPPER-CASE SYMBOL from self.board[x] (all the same)
            s = self.board[x][0].upper()
            self.board[x].remove(s)
            self.board[y].append(s)
            # add it to self.board[y]
            battles.append(y)
        atype, *aargs = lower_action
        if atype == "THROW":
            s, x = aargs
            self.board[x].append(s.lower())
            self.throws["lower"] += 1
            battles.append(x)
        else:
            x, y = aargs
            # remove ONE LOWER-CASE SYMBOL from self.board[x] (all the same)
            s = self.board[x][0].lower()
            self.board[x].remove(s)
            self.board[y].append(s)
            # add it to self.board[y]
            battles.append(y)
        # resolve hexes with new tokens:
        for x in battles:
            # TODO: include summary of battles in output?
            self.board[x] = _BATTLE(self.board[x])

        self._turn_detect_end()
        # TODO:
        # return a sanitised version of the action to avoid action injection?


    def _available_actions(self, colour):
        """
        A generator of currently-available actions for a particular player
        (assists validation).
        """
        throws = self.throws[colour]
        isplayer = str.islower if colour == "lower" else str.isupper
        if throws < 9:
            sign = -1 if colour == "lower" else 1
            throw_zone = (
                (r, q) for r, q in _SET_HEXES if sign * r >= 4 - throws
            )
            for x in throw_zone:
                for s in "rps":
                    yield "THROW", s, x
        occupied = {x for x, s in self.board.items() if any(map(isplayer, s))}
        for x in occupied:
            adjacent_x = _ADJACENT(x)
            for y in adjacent_x:
                yield "SLIDE", x, y
                if y in occupied:
                    opposite_y = _ADJACENT(y) - adjacent_x - {x}
                    for z in opposite_y:
                        yield "SWING", x, z

    def _turn_detect_end(self):
        """
        Register that a turn has passed: Update turn counts and detect
        termination conditions.
        """
        # register turn
        self.nturns += 1
        state = self._snap()
        self.history[state] += 1

        # analyse remaining tokens
        up_throws = 9 - self.throws["upper"]
        up_tokens = [
            s.lower() for x in self.board.values() for s in x if s.isupper()
        ]
        up_symset = set(up_tokens)
        lo_throws = 9 - self.throws["lower"]
        lo_tokens = [
            s for x in self.board.values() for s in x if s.islower()
        ]
        lo_symset = set(lo_tokens)
        up_invinc = [
            s for s in up_symset
            if (lo_throws == 0) and (_WHAT_BEATS[s] not in lo_symset)
        ]
        lo_invinc = [
            s for s in lo_symset
            if (up_throws == 0) and (_WHAT_BEATS[s] not in up_symset)
        ] 
        up_notoks = (up_throws == 0) and (len(up_tokens) == 0)
        lo_notoks = (lo_throws == 0) and (len(lo_tokens) == 0)
        up_onetok = (up_throws == 0) and (len(up_tokens) == 1)
        lo_onetok = (lo_throws == 0) and (len(lo_tokens) == 1)

        # condition 1: one player has no remaining throws or tokens
        if up_notoks and lo_notoks:
            self.result = "draw"
            return
        if up_notoks:
            self.result = "lower"
            return
        if lo_notoks:
            self.result = "upper"
            return

        # condition 2: both players have an invincible token
        if up_invinc and lo_invinc:
            self.result = "draw"
            return

        # condition 3: one player has an invincible token, the other has
        #              only one token remaining (not invincible by 2)
        if up_invinc and lo_onetok:
            self.result = "upper"
            return
        if lo_invinc and up_onetok:
            self.result = "lower"
            return

        # condition 4: the same state has occurred for a 3rd time
        if self.history[state] >= 3:
            self.result = "draw"
            return

        # condition 5: the players have had their 360th turn without end
        if self.nturns >= _MAX_TURNS:
            self.result = "draw"
            return

        # no conditions met, game continues
        return

    def _snap(self):
        """
        Capture the current board state in a hashable way
        (for repeated-state checking)
        """
        return (
            # same symbols/players tokens in the same positions
            tuple(
                (x, tuple(sorted(ts))) for x, ts in self.board.items() if ts
            ),
            # with the same number of throws remaining for each player
            self.throws["upper"],
            self.throws["lower"],
        )

    def over(self):
        """
        True iff the game has terminated.
        """
        return self.result is not None

    def end(self):
        """
        Conclude the game, extracting a string describing result (win or draw)
        This method should always be called to conclude a game so that this
        class has a chance to close the logfile, too.
        If the game is not over this is a no-op.
        """
        self.logger.info(self.result)
        return self.result

def play(weight):
    turn = 1

    player_1 = Player("upper", weight)
    player_2 = Player("lower", [2.12345008,1.76668476,0.63608363,1.76559664,0.14207212,0.03918779
,0.0202184,0.60314963,3.90086769,3.79006074]) #Greedy opponent, only escape from closest danger and capture closest target
    game = Game()
    while not game.over():
        action_1 = player_1.action()
        action_2 = player_2.action()
        game.update(action_1, action_2)
        player_1.update(opponent_action=action_2, player_action=action_1)
        player_2.update(opponent_action=action_1, player_action=action_2)
        turn+=1
    return player_1.state.evaluation_function([1, 1, 0.2, 0.5, 5, 0.2, 0.5, 5, 10, 10])
    
sol_per_pop = 15
num_weights = 10
pop_size = (sol_per_pop, num_weights)
new_population = []
for x in range(sol_per_pop - 8):
    new_population.append(np.array([np.random.uniform(0, 2), np.random.uniform(0, 3), np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 2), np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 2), np.random.uniform(0, 5), np.random.uniform(0, 5)]))
new_population.append(np.array([1.58345008,1.58668476,1.93608363,0.92559664,0.14207212,0.03918779
,0.0202184,1.76523969,4.21078375,4.35006074]))
new_population.append(np.array([1.58345008,1.58668476,1.93608363,0.92559664,1.23852716,0.03918779
,0.0202184,1.76523969,3.89078375,4.35006074]))
new_population.append(np.array([2.12345008,1.76668476,0.95608363,1.76559664,-0.73792788,0.03918779
,0.0202184,0.60314963,3.30086769,3.79006074]))
new_population.append(np.array([1.58345008,1.76668476,0.95608363,0.92559664,0.14207212,0.03918779
,0.0202184,1.76523969,3.90086769,3.79006074]))
new_population.append(np.array([2.12345008,1.76668476,0.95608363,0.92559664,1.47852716,0.03918779
,0.8002184,1.66523969,2.48086769,5.47006074]))
new_population.append(np.array([1.58345008,1.58668476,1.93608363,0.92559664,1.23852716,0.05918779
,0.0202184,1.76523969,3.89078375,4.35006074]))
new_population.append(np.array([2.12345008,1.58668476,1.93608363,1.76559664,1.23852716,0.03918779
,0.0202184,0.60314963,3.90086769,4.35006074]))
new_population.append(np.array([2.12345008,1.76668476,0.63608363,1.76559664,0.14207212,0.03918779
,0.0202184,0.60314963,3.90086769,3.79006074]))
new_population = np.array(new_population)
num_generations = 10
num_parents_mating = 5

"""
Code used here is modified from ahmedfgad: https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/master/Tutorial%20Project/ga.py
and TheAILearner: https://github.com/TheAILearner/Training-Snake-Game-With-Genetic-Algorithm/blob/master/Genetic_Algorithm.py
"""
def cal_pop_fitness(pop):
    fitness = []
    for i in range(pop.shape[0]):
        fit = play(pop[i].tolist())
        print(pop[i])
        print('fitness value of chromosome '+ str(i) +' :  ', fit)
        fitness.append(fit)
    return np.array(fitness)

def select_mating_pool(pop, fitness, num_parents):
    parents = np.empty((num_parents, pop.shape[1]))
    # print(parents)
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999
    return parents

def crossover(parents, offspring_size):
    # creating children for next generation 
    offspring = np.empty(offspring_size)
    
    for k in range(offspring_size[0]): 
  
        while True:
            parent1_idx = random.randint(0, parents.shape[0] - 1)
            parent2_idx = random.randint(0, parents.shape[0] - 1)
            # produce offspring from two parents if they are different
            if parent1_idx != parent2_idx:
                for j in range(offspring_size[1]):
                    if random.uniform(0, 1) < 0.5:
                        offspring[k, j] = parents[parent1_idx, j]
                    else:
                        offspring[k, j] = parents[parent2_idx, j]
                break
    return offspring

def mutation(offspring_crossover):
    # mutating the offsprings generated from crossover to maintain variation in the population
    
    for idx in range(offspring_crossover.shape[0]):
        for _ in range(25):
            i = randint(0,offspring_crossover.shape[1]-1)

        random_value = np.random.choice(np.arange(-1,1,step=0.02),size=(1),replace=False)
        offspring_crossover[idx, i] = offspring_crossover[idx, i] + random_value

    return offspring_crossover

start = time.time()

for generation in range(num_generations):
    print('##############        GENERATION ' + str(generation)+ '  ###############' )
    # Measuring the fitness of each chromosome in the population.
    fitness = cal_pop_fitness(new_population)
    print('#######  fittest chromosome in gneneration ' + str(generation) +' is having fitness value:  ', np.max(fitness))
    # Selecting the best parents in the population for mating.
    parents = select_mating_pool(new_population, fitness, num_parents_mating)

    # Generating next generation using crossover.
    offspring_crossover = crossover(parents, offspring_size=(pop_size[0] - parents.shape[0], num_weights))

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = mutation(offspring_crossover)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

end = time.time()

print(end-start)