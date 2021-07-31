
from ml_state import State
from gametheory import solve_game

import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

class Action:
    def __init__(self, action):
        self.action = action

class Player:

    

    def __init__(self, player, weight):
        if(player == "upper"):
            self.state = State.new(upper = True)
        else:
            self.state = State.new(upper = False)
        self.weight = weight
        self.player = player
        # print(self.state.get_upper_tokens())
        """
        Called once at the beginning of a game to initialise this player.
        Set up an internal representation of the game state.

        The parameter player is the string "upper" (if the instance will
        play as Upper), or the string "lower" (if the instance will play
        as Lower).
        """
        # put your code here

    def action(self):
        """
        Called at the beginning of each turn. Based on the current state
        of the game, select an action to play this turn.
        """
        # print(self.state.to_matrix())
        # actions_dict, probability = best_move(self.state, 0)

        a, p = best_move(self.state, 0, self.weight)
        
        choice = np.random.choice(len(a), 1, p = p[0])
        return a[choice[0]]

        # print(actions_dict, probability)
        # choice = np.random.choice(len(actions_dict), 1, probability)
        # print(choice[0])
        # return actions_dict[choice[0]]
        # np.random.choice(actions, 1, probability)
        # actions = self.state.actions_successors()
        # num = int(input())
        
        return actions[num]
        # put your code here
    
    def update(self, opponent_action, player_action):
        self.state = self.state.successor(player_action, opponent_action)
        # print(opponent_action)
        # print(player_action)
        """
        Called at the end of each turn to inform this player of both
        players' chosen actions. Update your internal representation
        of the game state.
        The parameter opponent_action is the opponent's chosen action,
        and player_action is this instance's latest chosen action.
        """
        # put your code here


def best_move(state, depth, weight):
    states = []
    terminal_depth = 0
    if(depth == terminal_depth):
        
        a, m = state.to_matrix(weight)
        return a, solve_game(m)


    a = state.actions()
    b = state.opponent_actions()
    for player_action in a:
        row = []
        for opponent_action in b:
            new_state = state.successor(player_action, opponent_action)
            row.append(best_move(new_state, depth + 1))
        states.append(row)
    return states
