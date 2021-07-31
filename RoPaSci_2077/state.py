import sys
import json
import math
import typing
import itertools
import collections
import random
import numpy as np

class Hex(typing.NamedTuple):
    """
    Hexagonal axial coordinates with basic operations and hexagonal
    manhatten distance.
    Thanks to https://www.redblobgames.com/grids/hexagons/ for some
    of the ideas implemented here.
    """
    r: int
    q: int

    @staticmethod
    def dist(x, y):
        """
        Hexagonal manhattan distance between two hex coordinates.
        """
        z_r = x.r - y.r
        z_q = x.q - y.q
        return (abs(z_r) + abs(z_q) + abs(z_r + z_q)) // 2

    def __add__(self, other):
        # this special method is called when two Hex objects are added with +
        return Hex(self.r + other[0], self.q + other[1])

HEX_RANGE = range(-4, +4+1)
ALL_HEXES = frozenset(
        Hex(r, q) for r in HEX_RANGE for q in HEX_RANGE if -r-q in HEX_RANGE
    )
HEX_STEPS = [Hex(r, q) for r, q in [(1,-1),(1,0),(0,1),(-1,1),(-1,0),(0,-1)]]


BEATS_WHAT = {'r': 's', 'p': 'r', 's': 'p'}
WHAT_BEATS = {'r': 'p', 'p': 's', 's': 'r'}

class Token(typing.NamedTuple):
    hex:    Hex
    symbol: str

class State(typing.NamedTuple):
    # Note: By subclassing namedtuple, we get efficient, immutable instances
    # and we automatically get sensible definitions for __eq__ and __hash__.

    # This class stores the game state in the format of two lists:
    # One holding the positions and symbols of all upper tokens:
    upper_tokens: tuple
    # And one for all the lower tokens:
    lower_tokens: tuple
    # There is also a set of valid hexes (all of those not blocked by
    # block tokens):
    all_hexes:    frozenset

    upper: bool

    num_throws: int
    opponent_num_throws: int
    

    # When subclassing namedtuple, we should control creation of instances
    # using a separate classmethod, rather than overriding __init__.
    @classmethod
    def new(cls, upper, upper_tokens = (), lower_tokens = (), all_hexes = ALL_HEXES, num_throws = 9, opponent_num_throws = 9):
        return cls(
                # TODO: Instead of sorted tuples, implement a frozen bag?
                upper_tokens=tuple(sorted(upper_tokens)),
                lower_tokens=tuple(sorted(lower_tokens)),
                all_hexes=all_hexes,
                upper=upper,
                num_throws=num_throws,
                opponent_num_throws=opponent_num_throws,
            )

    # Following the alternative constructor idiom, we'll create a separate
    # classmethod to allow creating our first state from the data dictionary.
    # @classmethod
    # def from_json(cls, file):
    #     data = json.load(file)
    #     upper_tokens = (Token(Hex(r, q), s) for s, r, q in data["upper"])
    #     lower_tokens = (Token(Hex(r, q), s) for s, r, q in data["lower"])
    #     all_hexes = ALL_HEXES - {Hex(r, q) for _s, r, q in data["block"]}
    #     return cls.new(upper_tokens, lower_tokens, all_hexes)


    # The core functionality of the state is to compute its available
    # actions and their corresponding successor states.
    def actions_successors(self):
        num = 0
        actions = self.actions()
        if(self.upper):
            print("Upper move available: ")
            for action in actions:
                print(action, num)
                num += 1
        else:
            print("Lower move available: ")
            for action in actions:
                print(action, num)
                num += 1
        return actions
            # yield action, self.successor(action)

    def actions(self):
        """
        Generate all available 'actions' (each 'action' is actually a
        collection of actions, one for each upper token).
        """
        if(self.upper):
            xs = [x for x, _s in self.upper_tokens]
            occupied_hexes = set(xs)
            def _adjacent(x):
                return self.all_hexes & {x + y for y in HEX_STEPS}
            def _token_actions(x):
                adjacent_x = _adjacent(x)
                for y in adjacent_x:
                    yield "SLIDE", x, y
                    if y in occupied_hexes:
                        opposite_y = _adjacent(y) - adjacent_x - {x}
                        for z in opposite_y:
                            yield "SWING", x, z
            result = []
            if(self.num_throws > 0):
                for rq in self.all_hexes:
                    if(self.num_throws - 5) <= rq.r:
                        for key in BEATS_WHAT.keys():
                            result.append(("THROW", key, rq))
            for token in (list(map(_token_actions, xs))):
                for action in token:
                    result.append(action)
            return tuple(result)
            
        else:
            xs = [x for x, _s in self.lower_tokens]
            occupied_hexes = set(xs)
            def _adjacent(x):
                return self.all_hexes & {x + y for y in HEX_STEPS}
            def _token_actions(x):
                adjacent_x = _adjacent(x)
                for y in adjacent_x:
                    yield "SLIDE", x, y
                    if y in occupied_hexes:
                        opposite_y = _adjacent(y) - adjacent_x - {x}
                        for z in opposite_y:
                            yield "SWING", x, z
            result = []
            if(self.num_throws > 0):
                for rq in self.all_hexes:
                    if(5 - self.num_throws) >= rq.r:
                        for key in BEATS_WHAT.keys():
                            result.append(("THROW", key, rq))
            for token in (list(map(_token_actions, xs))):
                for action in token:
                    result.append(action)
            
            return tuple(result)
            #result[0] = action type, result[1] = symbol/starting hex, result[2] = ending hex
                
            # return itertools.product(*map(_token_actions, xs))

    def opponent_actions(self):
        if(self.upper):
            xs = [x for x, _s in self.lower_tokens]
            occupied_hexes = set(xs)
            def _adjacent(x):
                return self.all_hexes & {x + y for y in HEX_STEPS}
            def _token_actions(x):
                adjacent_x = _adjacent(x)
                for y in adjacent_x:
                    yield "SLIDE", x, y
                    if y in occupied_hexes:
                        opposite_y = _adjacent(y) - adjacent_x - {x}
                        for z in opposite_y:
                            yield "SWING", x, z
            result = []
            if(self.opponent_num_throws > 0):
                for rq in self.all_hexes:
                    if(5 - self.opponent_num_throws) >= rq.r:
                        for key in BEATS_WHAT.keys():
                            result.append(("THROW", key, rq))
            for token in (list(map(_token_actions, xs))):
                for action in token:
                    result.append(action)
            return tuple(result)

        else:
            xs = [x for x, _s in self.upper_tokens]
            occupied_hexes = set(xs)
            def _adjacent(x):
                return self.all_hexes & {x + y for y in HEX_STEPS}
            def _token_actions(x):
                adjacent_x = _adjacent(x)
                for y in adjacent_x:
                    yield "SLIDE", x, y
                    if y in occupied_hexes:
                        opposite_y = _adjacent(y) - adjacent_x - {x}
                        for z in opposite_y:
                            yield "SWING", x, z
            result = []
            if(self.opponent_num_throws > 0):
                for rq in self.all_hexes:
                    if(self.opponent_num_throws - 5) <= rq.r:
                        for key in BEATS_WHAT.keys():
                            result.append(("THROW", key, rq))
            for token in (list(map(_token_actions, xs))):
                for action in token:
                    result.append(action)
            return tuple(result)
    
    def successor(self, player_action, opponent_action):
        if(self.upper):
            new_upper_tokens = list(self.upper_tokens)
            # print(player_action)
            # print(new_upper_tokens)
            new_lower_tokens = list(self.lower_tokens)
            new_num_throws = self.num_throws
            new_opponent_num_throws = self.opponent_num_throws
            new_hexes = set()
            if(player_action[0] == "THROW"):
                new_num_throws -= 1
                new_upper_tokens.append(Token(player_action[2], player_action[1]))
                old_upper_tokens_hexes = {t.hex for t in self.upper_tokens if t.hex != player_action[2]}
                new_hexes.add(player_action[2])
                new_hexes = new_hexes | old_upper_tokens_hexes
                
            else:
                tokens = [t for t in self.upper_tokens if t.hex == player_action[1]]
                token = tokens[0]
                new_upper_tokens.remove(token)
                new_upper_tokens.append(Token(player_action[2], token.symbol))
                old_upper_tokens_hexes = {t.hex for t in self.upper_tokens if t.hex != player_action[2]}
                new_hexes.add(player_action[2])
                new_hexes = new_hexes | old_upper_tokens_hexes
            
            if(opponent_action[0] == "THROW"):
                new_opponent_num_throws -= 1
                new_lower_tokens.append(Token(opponent_action[2], opponent_action[1]))
                old_lower_tokens_hexes = {t.hex for t in self.lower_tokens if t.hex != opponent_action[2]}
                new_hexes.add(opponent_action[2])
                new_hexes = new_hexes | old_lower_tokens_hexes
            else:
                tokens = [t for t in self.lower_tokens if t.hex == opponent_action[1]]
                token = tokens[0]
                new_lower_tokens.remove(token)
                new_lower_tokens.append(Token(opponent_action[2], token.symbol))
                old_lower_tokens_hexes = {t.hex for t in self.lower_tokens if t.hex != opponent_action[2]}
                new_hexes.add(opponent_action[2])
                new_hexes = new_hexes | old_lower_tokens_hexes
            # Battle
            safe_upper_tokens = []
            safe_lower_tokens = []
                        
            for x in new_hexes:
                ups_at_x = [t for t in new_upper_tokens  if t.hex == x]
                los_at_x = [t for t in new_lower_tokens if t.hex == x]
                symbols = {t.symbol for t in ups_at_x + los_at_x}
                if len(symbols) > 1:
                    for s in symbols:
                        p = BEATS_WHAT[s]
                        ups_at_x = [t for t in ups_at_x if t.symbol != p]
                        los_at_x = [t for t in los_at_x if t.symbol != p]
                safe_upper_tokens.extend(ups_at_x)
                safe_lower_tokens.extend(los_at_x)

            return self.new(True, safe_upper_tokens, safe_lower_tokens, self.all_hexes, new_num_throws, new_opponent_num_throws)
        else:
            new_upper_tokens = list(self.upper_tokens)
            new_lower_tokens = list(self.lower_tokens)
            new_num_throws = self.num_throws
            new_opponent_num_throws = self.opponent_num_throws
            new_hexes = set()
            if(player_action[0] == "THROW"):
                new_num_throws -= 1
                new_lower_tokens.append(Token(player_action[2], player_action[1]))
                old_lower_tokens_hexes = {t.hex for t in self.lower_tokens if t.hex != opponent_action[2]}
                new_hexes.add(player_action[2])
                new_hexes = new_hexes | old_lower_tokens_hexes

            else:
                tokens = [t for t in self.lower_tokens if t.hex == player_action[1]]
                token = tokens[0]
                new_lower_tokens.remove(token)
                new_lower_tokens.append(Token(player_action[2], token.symbol))
                old_lower_tokens_hexes = {t.hex for t in self.lower_tokens if t.hex != opponent_action[2]}
                new_hexes.add(player_action[2])
                new_hexes = new_hexes | old_lower_tokens_hexes
            
            if(opponent_action[0] == "THROW"):
                new_opponent_num_throws -= 1
                new_upper_tokens.append(Token(opponent_action[2], opponent_action[1]))
                old_upper_tokens_hexes = {t.hex for t in self.upper_tokens if t.hex != player_action[2]}
                new_hexes.add(opponent_action[2])
                new_hexes = new_hexes | old_upper_tokens_hexes
            else:
                tokens = [t for t in self.upper_tokens if t.hex == opponent_action[1]]
                token = tokens[0]
                new_upper_tokens.remove(token)
                new_upper_tokens.append(Token(opponent_action[2], token.symbol))
                old_upper_tokens_hexes = {t.hex for t in self.upper_tokens if t.hex != player_action[2]}
                new_hexes.add(opponent_action[2])
                new_hexes = new_hexes | old_upper_tokens_hexes
            
            # Battle
            safe_upper_tokens = []
            safe_lower_tokens = []
            for x in new_hexes:
                ups_at_x = [t for t in new_upper_tokens  if t.hex == x]
                los_at_x = [t for t in new_lower_tokens if t.hex == x]
                symbols = {t.symbol for t in ups_at_x + los_at_x}
                if len(symbols) > 1:
                    for s in symbols:
                        p = BEATS_WHAT[s]
                        ups_at_x = [t for t in ups_at_x if t.symbol != p]
                        los_at_x = [t for t in los_at_x if t.symbol != p]
                safe_upper_tokens.extend(ups_at_x)
                safe_lower_tokens.extend(los_at_x)

            return self.new(False, safe_upper_tokens, safe_lower_tokens, self.all_hexes, new_num_throws, new_opponent_num_throws)

    # For easier debugging, a helper method to print the current state.
    # def print(self, message="", **kwargs):
    #     board = collections.defaultdict(str)
    #     for t in self.upper_tokens:
    #         board[t.hex] += t.symbol.upper()
    #     for t in self.lower_tokens:
    #         board[t.hex] += t.symbol.lower()
    #     for x, s in board.items():
    #         board[x] = f"({s})"
    #     for x in ALL_HEXES - self.all_hexes:
    #         board[x] = "BLOCK"
    #     print_board(board, message, **kwargs)

    def get_lower_tokens(self):
        return self.lower_tokens

    def get_upper_tokens(self):
        return self.upper_tokens

    def is_upper(self):
        return self.upper

    def get_num_throws(self):
        return self.num_throws

    def get_opponent_num_throws(self):
        return self.opponent_num_throws


    def goal_test(self):

        # analyse remaining tokens
        up_throws = self.get_num_throws()
        up_tokens = [s.lower() for x in self.get_upper_tokens() for s in x.symbol]
        up_symset = set(up_tokens)
        lo_throws = self.get_opponent_num_throws()
        lo_tokens = [s.lower() for x in self.get_lower_tokens() for s in x.symbol]
        lo_symset = set(lo_tokens)
        up_invinc = [
            s for s in up_symset
            if (lo_throws == 0) and (WHAT_BEATS[s] not in lo_symset)
        ]
        lo_invinc = [
            s for s in lo_symset
            if (up_throws == 0) and (WHAT_BEATS[s] not in up_symset)
        ] 
        up_notoks = (up_throws == 0) and (len(up_tokens) == 0)
        lo_notoks = (lo_throws == 0) and (len(lo_tokens) == 0)
        up_onetok = (up_throws == 0) and (len(up_tokens) == 1)
        lo_onetok = (lo_throws == 0) and (len(lo_tokens) == 1)

        # condition 1: one player has no remaining throws or tokens
        if up_notoks and lo_notoks:
            return "draw"
        if up_notoks:
            return "lower"
        if lo_notoks:
            return "upper"

        # condition 2: both players have an invincible token
        if up_invinc and lo_invinc:
            return "draw"

        # condition 3: one player has an invincible token, the other has
        #              only one token remaining (not invincible by 2)
        if up_invinc and lo_onetok:
            return "upper"
        if lo_invinc and up_onetok:
            return "lower"

    def dist_from_capturing(self, max_dist):
        total_capture_distance = 0
        closest_capture_distance = max_dist + 1
        total_losing_distance = 0
        closest_losing_distance = max_dist + 1
        num_player_invincible_token = 0
        num_opponent_invincible_token = 0
        token_in_danger = 0
        opponent_token_in_danger = 0
        p_num = 0
        o_num = 0

        if(self.upper):
            for x, s in self.lower_tokens:
                r = WHAT_BEATS[s]
                ys = [y for y, r_ in self.upper_tokens if r_ == r] #list of upper tokens which can beat lower token x
                if ys:
                    for y in ys:
                        dist = Hex.dist(x, y)
                        if(dist == 1): #In immediate danger
                            opponent_token_in_danger += 1
                        total_capture_distance += Hex.dist(x, y)
                        p_num += 1
                        if(dist < closest_capture_distance):
                            closest_capture_distance = dist

                else:
                    num_opponent_invincible_token += 1
            
            for x, s in self.upper_tokens:
                r = WHAT_BEATS[s]
                ys = [y for y, r_ in self.lower_tokens if r_ == r]
                if ys:
                    for y in ys:
                        dist = Hex.dist(x, y)
                        if(dist == 1): #In immediate danger
                            token_in_danger += 1
                        total_losing_distance += Hex.dist(x, y)
                        o_num += 1
                        if(dist < closest_losing_distance):
                            closest_losing_distance = dist
                else:
                    num_player_invincible_token += 1
            return (total_capture_distance, p_num), closest_capture_distance, (total_losing_distance, o_num), closest_losing_distance, num_player_invincible_token, num_opponent_invincible_token, token_in_danger, opponent_token_in_danger
        else:
            for x, s in self.upper_tokens:
                r = WHAT_BEATS[s]
                ys = [y for y, r_ in self.lower_tokens if r_ == r] #list of upper tokens which can beat lower token x
                if ys:
                    for y in ys:
                        dist = Hex.dist(x, y)
                        if(dist == 1): #In immediate danger
                            opponent_token_in_danger += 1
                        total_capture_distance += Hex.dist(x, y)
                        p_num += 1
                        if(dist < closest_capture_distance):
                            closest_capture_distance = dist

                else:
                    num_opponent_invincible_token += 1
            
            for x, s in self.lower_tokens:
                r = WHAT_BEATS[s]
                ys = [y for y, r_ in self.upper_tokens if r_ == r]
                if ys:
                    for y in ys:
                        dist = Hex.dist(x, y)
                        if(dist == 1): #In immediate danger
                            token_in_danger += 1
                        total_losing_distance += Hex.dist(x, y)
                        o_num += 1
                        if(dist < closest_losing_distance):
                            closest_losing_distance = dist
                else:
                    num_player_invincible_token += 1

            return (total_capture_distance, p_num), closest_capture_distance, (total_losing_distance, o_num), closest_losing_distance, num_player_invincible_token, num_opponent_invincible_token, token_in_danger, opponent_token_in_danger
    
    # def check_invincible_tokens(self):
    #     if(self.upper_tokens)

    #new_state = state.successor(player_action, opponent_action)
    #evaluation_function(new_state)
    def evaluation_function(self):

        # features: (#self.upper_tokens, #self.lower_tokens), (num_throws, opponent_num_throws)
        #            (max_dist - total_capture_distance), (max_dist - closest_capture_distance), opponent_token_in_danger
        #            (max_dist - total_losing_distance), (max_dist - closest_losing_distance), token_in_danger, num_player_invincible_token, num_opponent_invincible_token

        if(self.upper):
            # player tokens = upper tokens
            # opponent tokens = lower tokens
            # Max distance from any point to any point on the board
            result = self.goal_test()
            if(result is not None):
                if(result == "draw"):
                    return 0
                if(result == "upper"):
                    return 200
                if(result == "lower"):
                    return -100
            w = [2.12345008,1.76668476,0.63608363,1.76559664,0.14207212,0.03918779
,0.0202184,0.60314963,3.90086769,3.79006074]
            max_dist = Hex.dist(Hex(-4, 4), Hex(4, -4))
            out_of_range = max_dist + 1

            dist = self.dist_from_capturing(max_dist)
            total_capture_distance = dist[0][0]
            x = dist[0][1]
            closest_capture_distance = dist[1]
            total_losing_distance = dist[2][0]
            y = dist[2][1]
            closest_losing_distance = dist[3]
            num_player_invincible_token = dist[4]
            num_opponent_invincible_token = dist[5]
            token_in_danger = dist[6]
            opponent_token_in_danger = dist[7]
            # num_actions = len(self.actions()) - len(self.opponent_actions()) #Take too much time

            if(closest_capture_distance == out_of_range and closest_losing_distance == out_of_range):
                return w[0]*(len(self.upper_tokens) - len(self.lower_tokens)) + w[1]*(self.num_throws - self.opponent_num_throws) + w[8]*num_player_invincible_token - w[9]*num_opponent_invincible_token
            elif(closest_capture_distance == out_of_range):
                return w[0]*(len(self.upper_tokens) - len(self.lower_tokens)) + w[1]*(self.num_throws - self.opponent_num_throws) - w[5]*(y*max_dist - total_losing_distance) - w[6]*(max_dist - closest_losing_distance) - w[7]*token_in_danger + w[8]*num_player_invincible_token - w[9]*num_opponent_invincible_token
            elif(closest_losing_distance == out_of_range):
                return w[0]*(len(self.upper_tokens) - len(self.lower_tokens)) + w[1]*(self.num_throws - self.opponent_num_throws) + w[2]*(x*max_dist - total_capture_distance) + w[3]*(max_dist - closest_capture_distance) + w[4]*opponent_token_in_danger + w[8]*num_player_invincible_token - w[9]*num_opponent_invincible_token

            return w[0]*(len(self.upper_tokens) - len(self.lower_tokens)) + w[1]*(self.num_throws - self.opponent_num_throws) + w[2]*(x*max_dist - total_capture_distance) + w[3]*(max_dist - closest_capture_distance) + w[4]*opponent_token_in_danger - w[5]*(y*max_dist - total_losing_distance) - w[6]*(max_dist - closest_losing_distance) - w[7]*token_in_danger + w[8]*num_player_invincible_token - w[9]*num_opponent_invincible_token

            # return random.randrange(-100, 100)

            #player/opponent invincible tokens
            #player tokens in danger
            #opponent tokens in danger
        else:
            # return 0 #Random opponent
            result = self.goal_test()
            if(result is not None):
                if(result == "draw"):
                    return 0
                if(result == "upper"):
                    return -100
                if(result == "lower"):
                    return 200
            # w = [30, 0, 0, 10, 10, 0, 10, 10, 20, 20] #Greedy opponent
            w = [2.12345008,1.76668476,0.63608363,1.76559664,0.14207212,0.03918779
,0.0202184,0.60314963,3.90086769,3.79006074]
            max_dist = Hex.dist(Hex(-4, 4), Hex(4, -4))
            out_of_range = max_dist + 1

            dist = self.dist_from_capturing(max_dist)
            total_capture_distance = dist[0][0]
            x = dist[0][1]
            closest_capture_distance = dist[1]
            total_losing_distance = dist[2][0]
            y = dist[2][1]
            closest_losing_distance = dist[3]
            num_player_invincible_token = dist[4]
            num_opponent_invincible_token = dist[5]
            token_in_danger = dist[6]
            opponent_token_in_danger = dist[7]
            # num_actions = len(self.actions()) - len(self.opponent_actions()) # Take too much time

            if(closest_capture_distance == out_of_range and closest_losing_distance == out_of_range):
                return w[0]*(len(self.upper_tokens) - len(self.lower_tokens)) + w[1]*(self.num_throws - self.opponent_num_throws) + w[8]*num_player_invincible_token - w[9]*num_opponent_invincible_token
            elif(closest_capture_distance == out_of_range):
                return w[0]*(len(self.upper_tokens) - len(self.lower_tokens)) + w[1]*(self.num_throws - self.opponent_num_throws) - w[5]*(y*max_dist - total_losing_distance) - w[6]*(max_dist - closest_losing_distance) - w[7]*token_in_danger + w[8]*num_player_invincible_token - w[9]*num_opponent_invincible_token
            elif(closest_losing_distance == out_of_range):
                return w[0]*(len(self.upper_tokens) - len(self.lower_tokens)) + w[1]*(self.num_throws - self.opponent_num_throws) + w[2]*(x*max_dist - total_capture_distance) + w[3]*(max_dist - closest_capture_distance) + w[4]*opponent_token_in_danger + w[8]*num_player_invincible_token - w[9]*num_opponent_invincible_token

            return w[0]*(len(self.upper_tokens) - len(self.lower_tokens)) + w[1]*(self.num_throws - self.opponent_num_throws) + w[2]*(x*max_dist - total_capture_distance) + w[3]*(max_dist - closest_capture_distance) + w[4]*opponent_token_in_danger - w[5]*(y*max_dist - total_losing_distance) - w[6]*(max_dist - closest_losing_distance) - w[7]*token_in_danger + w[8]*num_player_invincible_token - w[9]*num_opponent_invincible_token

    def to_matrix(self):
        actions_dict = {}
        matrix = []
        # eval_dict = {}
        num = 0
        count  = 0
        for player_action in self.actions():
            # eval_dict[num] = 0
            actions_dict[num] = player_action
            row = []
            for opponent_action in self.opponent_actions():
                new_state = self.successor(player_action, opponent_action)
                # print(player_action, opponent_action, new_state.evaluation_function())
                # eval_dict[num] += eval
                row.append(new_state.evaluation_function())
                count += 1
                # print(player_action, opponent_action)
            # eval_dict[num] = eval_dict[num]/count
            matrix.append(row)
            num += 1
            # print()
        # print("-------------------------")
        # return actions_dict, np.array(matrix), eval_dict
        return actions_dict, np.array(matrix)



    

