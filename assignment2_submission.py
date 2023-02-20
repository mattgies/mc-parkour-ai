import random

# i'm defining my tuning params here to ensure consistency
ALPHA = 0.2
GAMMA = 1.0
N = 10
EPSILON = 0.1

items=['pumpkin', 'sugar', 'egg', 'egg', "red_mushroom", "planks", "planks"]

food_recipes = {'pumpkin_pie': ['pumpkin', 'egg', 'sugar'],
                'pumpkin_seeds': ['pumpkin'],
                "bowl": ["planks", "planks"],
                "mushroom_stew": ["bowl", "red_mushroom"]}

rewards_map = {'pumpkin': -5, 'egg': -25, 'sugar': -10,
               'pumpkin_pie': 100, 'pumpkin_seeds': -50,
               "red_mushroom": 5, "planks": 5, "bowl": 1, "mushroom_stew": 100}

def is_solution(reward):
    return reward == 200

def get_curr_state(items):
    itemList = []
    for item, count in items:
        for _ in range(count):
            itemList.append(item)
    return tuple(sorted(itemList))

    # ORIGINAL:
    return len(items)

def choose_action(curr_state, possible_actions, eps, q_table):
    # eps = probability of random action
    # 1-eps = probability of best action

    # q_table = { state : { action1: qval1, ... } }

    epsThresh = random.random() # uniform sampling of 0.0 to 1.0 range
    if eps > epsThresh: # e.g. if 0.2 > 0.1, go with the random action
        a = random.randint(0, len(possible_actions) - 1)
        return possible_actions[a]
    else: # else, go with the best action according to the q_table
        actions_qvals = q_table[curr_state]
        best_q_val = max(actions_qvals.values())
        best_actions = [action for action, q_val in actions_qvals.items() if q_val == best_q_val]
        return random.choice(best_actions)


    # ORIGINAL:
    rnd = random.random()
    a = random.randint(0, len(possible_actions) - 1)
    return possible_actions[a]
