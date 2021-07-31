
from RoPaSci_2077.state import State
from RoPaSci_2077.gametheory import solve_game

import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)
# np.random.seed(0)
class Action:
    def __init__(self, action):
        self.action = action

class Player:

    

    def __init__(self, player):
        if(player == "upper"):
            self.state = State.new(upper = True)
        else:
            self.state = State.new(upper = False)
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
        try:
            a, p= best_move(self.state, 0)
            # print(m)
            sum = 0
            avg = 0
            count=0
            num=0
            # Debug: print each action with corresponding probability and average pay-off matrix's value in corresponding row
            # for action in a:
            #     print(a[action], p[0][action], e[action])
            #     sum+=p[0][action]
            # print(sum)
            choice = np.random.choice(len(a), 1, p = p[0])
            return a[choice[0]]
        except ValueError:
            choice = np.random.choice(len(a), 1)    #Due to unknown error from gametheory (Optimisation error exception)
            return a[choice[0]]                     #We choose a random action when that exceeption occur
        except TypeError:
            choice = np.random.choice(len(a), 1)
            return a[choice[0]]
        # print(actions_dict, probability)
        # choice = np.random.choice(len(actions_dict), 1, probability)
        # print(choice[0])
        # return actions_dict[choice[0]]
        # np.random.choice(actions, 1, probability)
        # actions = self.state.actions_successors()
        # num = int(input())
        
        # return actions[num]
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


def best_move(state, depth):
    # states = []
    terminal_depth = 0
    if(depth == terminal_depth):
        
        a, m= state.to_matrix()
        return a, solve_game(m)


    # a = state.actions()
    # b = state.opponent_actions()
    # for player_action in a:
    #     row = []
    #     for opponent_action in b:
    #         new_state = state.successor(player_action, opponent_action)
    #         row.append(best_move(new_state, depth + 1))
    #     states.append(row)
    # return states
