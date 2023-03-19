import numpy as np
import math
import MalmoPython
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import sys
import time
import random

import xmlgen
import parkourcourse1 as course
import observationgrid1 as obsgrid
from worldClasses import *
import matplotlib.pyplot as plt

from fp_plotting import Plotter


# METRICS
loss_function_returns = []
episode_rewards = []
episode_reward_running_avgs = []



# CONSTANTS
MAX_HISTORY_LENGTH = 50000
MAX_ACTIONS_PER_EPISODE = 10000
UPDATE_MODEL_AFTER_N_FRAMES = 4
UPDATE_TARGET_AFTER_N_FRAMES = 100
NUM_EPISODES = 200
AVERAGE_REWARD_NEEDED_TO_END = 500
BATCH_SIZE = 25

GROUNDED_DISTANCE_THRESHOLD = 0.2 # The highest distance above a block for which the agent is considered to be stepping on it.

# training parameters
EPSILON_START = EPSILON = 0.6
EPSILON_MIN = 0.1
GAMMA = 0.99
ALPHA = 0.7 # = [0,1], How much prioritization is used, with 0 being no prioritization
BETA = 0.5 # = [0,1], Annealing factor (I don't know what this is)
optimizer = keras.optimizers.Adam(learning_rate=0.0025)
# loss_function = keras.losses.Huber()
def loss_function(arr1, arr2):
    # print("array lengths:", len(arr1), len(arr2))
    # Q: over time, loss always increases to a massive magnitude (both pos and neg values) but array lengths are always BATCH_SIZE (which is correct)
    # so why is loss getting so big?
    sum = 0
    i = 0
    while i < len(arr1) and i < len(arr2):
        sum += (arr1[i] - arr2[i])
        i += 1
    return sum
loss_function = keras.losses.Huber()


# states (state space is nearly infinite; not directly defined in our code)
# actions
actionNamesToActionsMap: dict() = {
    "stopMove": "move 0.0",
    "moveHalf": "move 0.5",
    "moveFull": "move 1.0",
    "moveBackwards": "move -0.5",
    "jumpFull": "jump 1",
    "stopJump": "jump 0",
    "turnRight": "turn 1.0",
    "turnLeft": "turn -1.0",
    "stopTurn": "turn 0.0",
    "keepSameActions": ""
}
actionNames: list() = list(actionNamesToActionsMap.keys())
NUM_ACTIONS: int = len(actionNames)

# rewards
rewardsMap: dict() = {
    "steppedOnPreviouslySeenBlock": 0, # -0.0001, # 0s, # -0.2,
    "newBlockSteppedOn": 10,
    "death": 0, # -1,
    "goalReached": 1000
}


# replay values
# past_states = []
# past_next_states = []
# past_actions = []
# past_rewards = []
# past_done = []
# replay_memory = []



past_states = []
past_next_states = []
past_actions = []
past_rewards = []
past_done = []
past_priorities = []
past_isw = []

episode_reward = 0
frame_number = 0
episode_number = 0




# models
def create_model():
    input_shape = (11,) # 3 for closest block, 3 for velocity vector, 2 for two boolean inputs

    inputs = layers.Input(shape=input_shape)

    layer1 = layers.Dense(16, activation='relu', name="layer1")(inputs)
    layer2 = layers.Dense(16, activation='relu', name="layer2")(layer1)
    layer3 = layers.Dense(16, activation='relu', name="layer3")(layer2)

    action = layers.Dense(NUM_ACTIONS, activation='linear')(layer3)

    # return keras.Model(inputs=inputs, outputs=action)
    model = keras.Model(inputs=inputs, outputs=action)
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.0025))
    return model

model = create_model()
target_model = create_model()





# loss_func = tf.keras.losses.MSE()

# global agent variables
prev_agent_position = Vector(0.5, 227.0, 0.5) # Where the player was last update
prev_agent_yaw = 0
prev_block_below_agent = Block(0,0,0,"air") # What block was below the player last update
blocks_walked_on = set()
## Used for testing prints
reward_of_all_episodes = 0
episodes_that_succeeded = []

# functions
def GetMissionXML(summary=""):
    return xmlgen.XMLGenerator(
        cube_coords=course.CUBE_COORDS,
        observation_grids=obsgrid.OBSERVATION_GRIDS,
        goal_coords=course.GOAL_COORDS,
        goal_reward=rewardsMap["goalReached"]
    )

def get_nearby_walkable_blocks(observations):
    """
    Returns list of blocks near the agent that can be walked on
    (not air or lava)

    returns: list of Block
    """
    grid = observations.get(u'floor5x5x5')
    
    # NOTE showed that "grid" starts off as an array of just "air" for the first few frames, then updates with actual info
    # this is probably due to the map needing to load in from the XML, but the agent is alive before the map is fully loaded
    # print("\n", len(grid))
    # print(grid)
    player_location = [int(observations[u'XPos']), int(observations[u'YPos']), int(observations[u'ZPos'])]
    result = []
    # TODO: Make these variables
    i = 0
    for y in range(-2, 2 + 1):
        for z in range(-2, 2 + 1):
            for x in range(-2, 2 + 1):
                # NOTE showed that (0,-1,0) within the grid is the location of the block beneath the player
                # if (x == 0 and y == -1 and z == 0):
                #     print(grid[i])
                if grid[i] != "air" and grid[i] != "lava":
                    result.append(Block(player_location[0]+x, player_location[1]+y, player_location[2]+z, grid[i]))
                i += 1
    # if len(result) == 0:
    #     print("ERROR LINE 161", grid)
    return result


def get_block_below_agent(observations):
    """
    Get an object representing the block beneath the agent's position.
    Does not count the block that the agent is occupying right now.

    returns: Block
    """
    player_location = [int(observations[u'XPos']), int(observations[u'YPos']), int(observations[u'ZPos'])]
    
    grid = observations.get(u'floor5x5x5')
    i = 0
    for y in range(-2, 2 + 1):
        for z in range(-2, 2 + 1):
            for x in range(-2, 2 + 1):
                # NOTE showed that (0,-1,0) within the grid is the location of the block beneath the player
                if (x == 0 and y == -1 and z == 0):
                    return Block(player_location[0], player_location[1]-1, player_location[2], grid[i])
                i += 1



def is_grounded(observations):
    """
    returns: bool: true if touching ground
    """
    grid = observations.get(u'floor5x5x5')  
    player_height = float(observations[u'YPos'])
    player_height_rounded = int(player_height)
    block_name_below_player = get_block_below_agent(observations).name # grid[5 * int(5 / 2) + int(5 / 2)] 
    return block_name_below_player != "lava" and block_name_below_player != "air" and (abs(player_height - player_height_rounded) <= GROUNDED_DISTANCE_THRESHOLD)


def format_state(raw_state) -> "tuple(float, float, float, float, float, float, float, bool)":
    """
    Computes state of the agent relative to the world for this update.
    Return order:
    direction to nearest unwalked block: x, y, z
    movement direction of agent: dx, dy, dz
    yaw
    is_grounded

    returns: tuple of 7 floats and 1 bool
    """

    global prev_agent_position
    global blocks_walked_on

    # TODO: Could do try/catch and skip this loop if it gives an out of bounds exception.
    # NOTE: Getting the observations multiple times on the same frame most likely causes the out of bounds exception.
    # Get agent observations for this update.

    # try:
    obs_text = raw_state.observations[-1].text
    # except IndexError as err:
    #     raise ValueError("Unable to get new agent state.")

    obs = json.loads(obs_text) # most recent observation

    # Where agent is this update.
    agent_position = Vector(float(obs[u'XPos']), float(obs[u'YPos']), float(obs[u'ZPos']))
    agent_position_int = Vector(int(obs[u'XPos']), int(obs[u'YPos']), int(obs[u'ZPos']))

    # Grounded check
    grounded_this_update = is_grounded(obs)

    # Get grid observations
    blocks = get_nearby_walkable_blocks(obs)
    direction_to_closest_unwalked_block = Vector(0,0,0)
    closest_block_distance = 10 ** 4
    for b in blocks:
        if b not in blocks_walked_on:
            direction = b.position() - agent_position
            if direction.magnitude() < closest_block_distance:
                direction_to_closest_unwalked_block = direction
                closest_block_distance = direction.magnitude()
    # NOTE proved that direction_to_closest_unwalked_block is only ever (0,0,0) when blocks is empty
    # ergo, there's an issue with get_nearby_walkable_blocks
    # TODO check why this is 0 when you land on a new block
    # if direction_to_closest_unwalked_block.x == 0:
    #     print(blocks)

    # Velocity vector
    velocity = agent_position - prev_agent_position
    prev_agent_position = agent_position

    # Facing direction. Doesn't need to look up or down
    yaw = obs[u'Yaw']
    global prev_agent_yaw
    yawChange = yaw - prev_agent_yaw
    prev_agent_yaw = yaw

    # distance from edge of block
    # Y axis is up, so only care about X and Z
    agent_pos_with_zeroed_y = Vector(agent_position.x, 0, agent_position.z)
    edge1 = Vector(math.ceil(agent_position.x), 0, math.ceil(agent_position.z))
    edge2 = Vector(math.ceil(agent_position.x), 0, math.floor(agent_position.z))
    edge3 = Vector(math.floor(agent_position.x), 0, math.ceil(agent_position.z))
    edge4 = Vector(math.floor(agent_position.x), 0, math.floor(agent_position.z))
    edges = [edge1, edge2, edge3, edge4]
    distance_from_edge_of_cur_block = 10 ** 4
    closest_edge = None
    for e in edges:
        dist_from_e = (e - agent_pos_with_zeroed_y).magnitude()
        if dist_from_e < distance_from_edge_of_cur_block:
            distance_from_edge_of_cur_block = dist_from_e
            closest_edge = e
    # print("Agent position", agent_position)
    # print("Closest edge", closest_edge)
    # print("Distance from edge", distance_from_edge_of_cur_block)

    return (
        direction_to_closest_unwalked_block.x,
        direction_to_closest_unwalked_block.y,
        direction_to_closest_unwalked_block.z,
        direction_to_closest_unwalked_block.magnitude(),
        distance_from_edge_of_cur_block,
        velocity.x,
        velocity.y,
        velocity.z,
        yaw,
        yawChange,
        grounded_this_update
    )


def choose_action(model, state):
    """
    Called once per frame, to determine the next action to take given the current state
    Uses the value of epsilon to determine whether to choose a random action or the best action (via tf.argmax)
    
    if we want: update epsilon to decay toward its minimum
    """
    
    if np.random.rand(1)[0] < EPSILON:
        return np.random.choice(NUM_ACTIONS)
    else:
        action_probs = model(tf.expand_dims(tf.convert_to_tensor(state), 0), training=False)
        # print("past rewards", past_rewards)
        # print("new probs")
        # for k in range(len(actionNames)):
        #     print(actionNames[k], float(action_probs[0][k]))

        # print("\n\n\n\n")
        action = tf.argmax(action_probs[0]).numpy()
        return action


def obs_is_valid(raw_state):
    obs_text = raw_state.observations[-1].text
    obs = json.loads(obs_text)
    if not u'XPos' in obs \
        or not u'YPos' in obs \
        or not u'ZPos' in obs \
        or not u'Yaw' in obs \
        or not u'floor5x5x5' in obs:
        return False
    return True


def take_action(action, agent_host):
    """
    Called once per frame, after choose_action
    
    returns: void, just runs the action
    """
    agent_host.sendCommand(actionNamesToActionsMap[actionNames[action]])


def calculate_reward(raw_state):
    """
    Called once per frame, after new state
    
    returns: integer representing reward
    """
    reward = 0

    global blocks_walked_on
    global prev_block_below_agent
    world_state = raw_state

    obs_text = world_state.observations[-1].text
    obs = json.loads(obs_text) # most recent observation

    # need to update, but very rudimentary reward checking system
    # check for game finished
    grid = obs.get(u'floor5x5x5')  
    player_height = float(obs[u'YPos'])

    player_height_rounded = int(player_height)
    block_name_below_player = get_block_below_agent(obs).name
    
    agent_position_int = Vector(int(obs[u'XPos']), int(obs[u'YPos']), int(obs[u'ZPos']))
        
    grounded_this_update = is_grounded(obs)


    # TODO: Trying to see if we were grounded in between this update and the previous.
    prev_distance_above_block = prev_agent_position.y - int(prev_agent_position.y)
    cur_distance_above_block = float(obs[u'YPos']) - agent_position_int.y
    y_position_delta_upper_bound = 0.45
    grounded_between_updates = False
    if ((block_name_below_player == "stone" or prev_block_below_agent.name == "stone") and
        (GROUNDED_DISTANCE_THRESHOLD < prev_distance_above_block < y_position_delta_upper_bound) and 
        (GROUNDED_DISTANCE_THRESHOLD < cur_distance_above_block < y_position_delta_upper_bound)):
        # For this update and the previous, the agent was hovering above the walkable block below it.
        # We assume that they must have touched the ground between the updates.
        # TODO: Simplify to one line
        grounded_between_updates = True

    if (not grounded_this_update and not grounded_between_updates):
        return 0

    block_stepping_on = get_block_below_agent(obs)
    if (grounded_between_updates and prev_block_below_agent.name != "air" and prev_block_below_agent.name != "lava" 
        and prev_block_below_agent not in blocks_walked_on):
        # We were on a block between this update and the last. We only recorded that block from the previous update,
        # so use the stored value when adding which blocks we've stepped on.
        print("Stepped on a new block:", prev_block_below_agent.name, prev_block_below_agent.position(), "last update")
        blocks_walked_on.add(prev_block_below_agent)
        return rewardsMap["newBlockSteppedOn"]
    if block_stepping_on.name != "air" and block_stepping_on.name != "lava":
        # We have found the block we're stepping on, if any.
        # See if we have stepped on it before.
        if block_stepping_on.name != "stone":
            return 0
        elif block_stepping_on in blocks_walked_on:
            return rewardsMap["steppedOnPreviouslySeenBlock"]
        else:
            # TODO: Testing print to see when Agent thinks it's on a new block
            print("Stepped on a new block:", block_stepping_on.name, block_stepping_on.position())
            blocks_walked_on.add(block_stepping_on)
            return rewardsMap["newBlockSteppedOn"]

    # for b in blocks:
    #     if agent_position_int == b.position() + Vector(0,1,0):
    #         # We have found the block we're stepping on, if any.
    #         # See if we have stepped on it before.
    #         if b.name != "stone":
    #             return 0, False
    #         elif b in blocks_walked_on:
    #             return rewardsMap["steppedOnPreviouslySeenBlock"], False
    #         else:
    #             # TODO: Testing print to see when Agent thinks it's on a new block
    #             print("\nStepped on a new block:", b.name, b.position())
    #             blocks_walked_on.add(b)
    #         break
    return 0


def update_target_model():
    """
    """
    target_model.set_weights(model.get_weights())


def add_entry_to_replay(state, next_state, action, reward, done, priority):
    """
    Called every time the AI takes an action
    Updates the replay buffers in place

    returns: void
    """
    # replay_memory.append((state, next_state, action, reward, done, priority, 1.0))
    past_states.append(state)
    past_next_states.append(next_state)
    past_actions.append(action)
    past_rewards.append(reward)
    past_done.append(done)
    past_priorities.append(priority)
    past_isw.append(1.0)


def remove_first_entry_in_replay():
    """
    This function will be called when our replay buffers are longer than MAX_HISTORY_LENGTH
    """
    # del replay_memory[0]
    del past_states[0]
    del past_next_states[0]
    del past_actions[0]
    del past_rewards[0]
    del past_done[0]
    del past_priorities[0]
    del past_isw[0]


def training_loop(agent_host):
    global blocks_walked_on 
    blocks_walked_on.clear()
    episode_reward = 0
    episode_done = False
    frame_number = 0
    cur_state_raw = agent_host.getWorldState()
    episode_start_time = time.time()
    has_stepped_on_first_block = False
    model.set_weights(target_model.get_weights())

    while (len(cur_state_raw.observations) == 0) or (not obs_is_valid(cur_state_raw)):
        cur_state_raw = agent_host.getWorldState()
    cur_state = format_state(cur_state_raw)

    for _ in range(MAX_ACTIONS_PER_EPISODE):
        # if len(past_states) > 0:
        #     cur_state = past_next_states[-1]
        if len(past_states) > 0:
            last_next_state = past_next_states[-1]
            cur_state = last_next_state
        action = choose_action(model, cur_state)
        # if action != actionNames.index("keepSameActions"):
        #     agent_host.sendCommand("move 0")
        #     agent_host.sendCommand("turn 0")
        #     agent_host.sendCommand("jump 0")
        # else:
        #     print("keeping same actions")

        take_action(action, agent_host)

        # time.sleep(0.05)

        goal_reached = False
        is_dead = False
        next_state_raw = agent_host.getWorldState()
        while (len(next_state_raw.observations) == 0) or (not obs_is_valid(next_state_raw)):
            next_state_raw = agent_host.getWorldState()
            if (not next_state_raw.is_mission_running):
                break
        frame_number += 1
        
        if (not next_state_raw.is_mission_running):
            next_state = cur_state # TODO examine whether this is valid
            # TODO: Hack to check if player has reached the goal or not. There are multiple ways to end the mission
            if(next_state_raw.rewards):
            # if cur_state[7]: # ISSUE: if player is jumping onto the diamond block, the mission ends but the reward is not given. it's only given via this condition if the agent walks onto the goal block
                goal_reached = True
            else:
                # Agent has died or fallen off the map
                is_dead = True

        
        episode_time_taken = time.time() - episode_start_time
        if is_dead:
            reward = rewardsMap["death"] / episode_time_taken
            episode_done = True
        elif not goal_reached:
            next_state = format_state(next_state_raw)
            # reward = 0 if len(next_state_raw.rewards) == 0 else next_state_raw.rewards[-1].getValue()
            reward = calculate_reward(next_state_raw) / episode_time_taken
            episode_done = False
            # Remember previous block. Used in calculate_reward
            global prev_block_below_agent
            prev_block_below_agent = get_block_below_agent(json.loads(next_state_raw.observations[-1].text))
        else:
            # reward = 0 if len(next_state_raw.rewards) == 0 else next_state_raw.rewards[-1].getValue()
            reward = rewardsMap["goalReached"] / episode_time_taken
            episode_done = True

        if reward != 0 and not has_stepped_on_first_block:
            reward = 0
            has_stepped_on_first_block = True

        # NOTE: proved that blocks_walked_on updates correctly (at least for a test of first block and jumping to the next block)
        # print(blocks_walked_on)
        # NOTE: proved there is an issue with the way distance_to_nearest_unwalked_block is calculated
        # if next_state[0] == 0:
        #     print("IS_DEAD:", is_dead, "GOAL_REACHED:", goal_reached)
        #     print("CUR STATE:", cur_state)
        #     print("NEXT STATE:", next_state)
        #     print("\n\n")
        if len(past_states) <= 1:
            priority = 1.0
        else:
            priority = max(past_priorities)

        episode_reward += reward
        add_entry_to_replay(state=cur_state, next_state=next_state, action=action, reward=reward, done=episode_done, priority=priority)
        
        if frame_number % UPDATE_MODEL_AFTER_N_FRAMES == 0 and frame_number > BATCH_SIZE:
            normalized_probs = np.divide(past_priorities, np.sum(past_priorities))
            sampled_transitions = np.random.choice(range(len(past_states)), size=BATCH_SIZE, replace=False, p=normalized_probs)

            sampled_states = np.array([past_states[i] for i in sampled_transitions])
            sampled_next_states = np.array([past_next_states[i] for i in sampled_transitions])
            sampled_actions = np.array([past_actions[i] for i in sampled_transitions])
            sampled_rewards = np.array([past_rewards[i] for i in sampled_transitions])
            sampled_priorities = np.array([past_priorities[i] for i in sampled_transitions])

            predicted_future_rewards = target_model.predict(sampled_next_states, verbose=0) # prints to stdout
            bellman_updated_q_vals = sampled_rewards + GAMMA * tf.reduce_max(predicted_future_rewards, axis=1)
            N = len(past_states)
            for j in range(len(sampled_transitions)):
                original_index = sampled_transitions[j]
                max_isw = max(past_isw)
                p_j_alpha = sampled_priorities[j] ** ALPHA
                p_alpha_sum = sum([p ** ALPHA for p in past_priorities])
                isw = ((N * p_j_alpha / p_alpha_sum) ** (-BETA)) / max_isw # line 10
                td_error = bellman_updated_q_vals[j] - past_rewards[original_index - 1] # line 11
                past_priorities[original_index] = abs(td_error) # line 12

                # line 13
                action_mask = tf.one_hot(sampled_actions, NUM_ACTIONS)
                with tf.GradientTape() as tape:
                    original_q_vals = model(sampled_states)
                    original_q_vals_for_actions = tf.reduce_sum(tf.multiply(original_q_vals, action_mask), axis=1)
                    expected, actual = [bellman_updated_q_vals[j]], [original_q_vals_for_actions[j]]
                    loss = loss_function(expected, actual)
                    loss_function_returns.append(loss)

                grads = tape.gradient(loss, model.trainable_variables)
                grads = [isw * td_error * grad for grad in grads]
                optimizer.apply_gradients(zip(grads, model.trainable_variables))










            # random_transitions = random.sample(replay_memory, k=BATCH_SIZE) # samples without replacement
            

            # # random_indices = np.random.choice(range(len(past_states)), size=BATCH_SIZE)
            # sampled_states = np.array([t[0] for t in random_transitions])
            # sampled_next_states = np.array([t[1] for t in random_transitions])
            # sampled_actions = np.array([t[2] for t in random_transitions])
            # sampled_rewards = np.array([t[3] for t in random_transitions])
            # sampled_priorities = np.array([t[-2] for t in random_transitions])

            # past_priorities = np.array([t[-2] for t in replay_memory])
            # past_rewards = np.array([t[3] for t in replay_memory])
            # past_isw = np.array([tr[-1] for tr in replay_memory])

            # predicted_future_rewards = target_model.predict(sampled_next_states, verbose=0) # prints to stdout
            # bellman_updated_q_vals = sampled_rewards + GAMMA * tf.reduce_max(predicted_future_rewards, axis=1)
            # N = len(replay_memory)
            # for j in range(len(random_transitions)):
            #     original_index = random_transitions[j]
            #     max_isw = max(past_isw)
            #     p_j_alpha = sampled_priorities[j] ** ALPHA
            #     p_alpha_sum = sum([p ** ALPHA for p in past_priorities])
            #     isw = ((N * p_j_alpha / p_alpha_sum) ** (-BETA)) / max_isw # line 10
            #     td_error = bellman_updated_q_vals[j] - past_rewards[original_index - 1] # line 11
            #     past_priorities[original_index] = abs(td_error) # line 12

            #     # line 13
            #     action_mask = tf.one_hot(sampled_actions, NUM_ACTIONS)
            #     with tf.GradientTape() as tape:
            #         original_q_vals = model(sampled_states)
            #         original_q_vals_for_actions = tf.reduce_sum(tf.multiply(original_q_vals, action_mask), axis=1)
            #         expected, actual = [bellman_updated_q_vals[j]], [original_q_vals_for_actions[j]]
            #         loss = loss_function(expected, actual)
            #         loss_function_returns.append(loss)

            #     grads = tape.gradient(loss, model.trainable_variables)
            #     grads = [isw * td_error * grad for grad in grads]
            #     optimizer.apply_gradients(zip(grads, model.trainable_variables))
            





            # # next_state = tf.convert_to_tensor(next_state)
            # predicted_future_rewards = target_model.predict(sampled_next_states, verbose=0) # prints to stdout
            # bellman_updated_q_vals = sampled_rewards + GAMMA * tf.reduce_max(predicted_future_rewards, axis=1)
            # action_mask = tf.one_hot(sampled_actions, NUM_ACTIONS)

            # with tf.GradientTape() as tape:
            #     print(sampled_states)
            #     original_q_vals = model(sampled_states)
            #     original_q_vals_for_actions = tf.reduce_sum(tf.multiply(original_q_vals, action_mask), axis=1)
            #     # print(bellman_updated_q_vals, original_q_vals_for_actions)
            #     # R + GAMMA * Q(s',a')
            #     loss = loss_function(bellman_updated_q_vals, original_q_vals_for_actions)
            #     # print("Loss:", loss)
            #     loss_function_returns.append(loss)
            
            # gradients = tape.gradient(loss, model.trainable_variables)
            # optimizer.apply_gradients(zip(gradients, model.trainable_variables))


            # q_s_values = model(sampled_states)
            # future_predictions = target_model.predict(sampled_next_states, verbose=0)
            # k = 0
            # for tr in random_transitions:
            #     state, next_state, action, reward, done = tr

            #     q_s = q_s_values[k]
            #     target = reward + GAMMA * np.amax(future_predictions[k])
            #     print("target:", target)
            #     target_f = q_s
            #     print("target_f:", target_f)
            #     target_f_list = [n.numpy() for n in target_f]
            #     print("target f list", target_f_list)
            #     target_f_list[action] = target
            #     target_f = tf.constant(target_f_list)
            #     model.fit(state, target_f, epochs=1, verbose=1)

            #     # with tf.GradientTape() as tape:
            #     #     q_s_a = q_s_values[k][action].numpy()
            #     #     print("Q(s,a) =", q_s_a)
            #     #     q_sprime_aprime = max(future_predictions[k])
            #     #     target = reward + GAMMA * q_sprime_aprime
            #     #     print("target =", target)
            #     #     loss = loss_function(target, q_s_a)
            #     #     loss_function_returns.append(loss)
                
            #     # gradients = tape.gradient(loss, model.trainable_variables)
            #     # optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            #     k += 1



        # if ready to update model (% UPDATE_MODEL_AFTER_N_FRAMES == 0)
        # take sample from replay buffers & update q-values
        # update value of episode_done
        # if frame_number % UPDATE_TARGET_AFTER_N_FRAMES == 0:
        #     # if ready to update target model
        #     update_target_model()


        if len(past_states) > MAX_HISTORY_LENGTH:
            remove_first_entry_in_replay()

        if episode_done:
            # Add values to testing prints
            global reward_of_all_episodes
            global episodes_that_succeeded
            global i
            reward_of_all_episodes += episode_reward
            episode_rewards.append(episode_reward)
            episode_reward_running_avgs.append(reward_of_all_episodes / (i+1))

            # update_target_model()
            target_model.set_weights(model.get_weights())
            if goal_reached:
                episodes_that_succeeded.append(i+1) # adding 1 since it's 0-indexed
            print("Episode reward:", episode_reward, "Average reward:", (reward_of_all_episodes / (i+1)), "Successful episodes:", episodes_that_succeeded, "\n")

            break


# Create default Malmo objects:
my_client_pool = MalmoPython.ClientPool()
my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

my_mission = MalmoPython.MissionSpec(GetMissionXML(), True)

# Attempt to start a mission:
max_retries = 3
for i in range(NUM_EPISODES):
    # if reward_of_all_episodes / (i+1) > AVERAGE_REWARD_NEEDED_TO_END:
    #     print("AI too good")
        # break

    print("Episode " + str(i+1) + " of " + str(NUM_EPISODES))
    
    my_mission_record = MalmoPython.MissionRecordSpec()

    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_client_pool, my_mission_record, 0, "myExperimentString" )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2)

    # Loop until mission starts:
    print("Waiting for the mission to start ", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)

    print()
    print("Mission running")

    # testing training loop function
    training_loop(agent_host)
    time.sleep(0.5)

    # decay epsilon (model should be more confident)
    # TODO alter this to better suit the realistic learning time
    EPSILON -= ((EPSILON_START - EPSILON_MIN) / NUM_EPISODES)


plotter = Plotter(NUM_EPISODES, loss_function_returns, episode_rewards, episode_reward_running_avgs)
plotter.plot()
