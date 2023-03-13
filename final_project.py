from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object
import numpy as np
import math
import MalmoPython
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import copy
import os
import sys
import time

import xmlgen
import parkourcourse1 as course
import observationgrid1 as obsgrid
from worldClasses import *
import matplotlib.pyplot as plt


# METRICS
loss_function_returns = []
episode_rewards = []
episode_reward_running_avgs = []



# CONSTANTS
MAX_HISTORY_LENGTH = 50000
MAX_ACTIONS_PER_EPISODE = 10000
UPDATE_MODEL_AFTER_N_FRAMES = 5
UPDATE_TARGET_AFTER_N_FRAMES = 100
NUM_EPISODES = 200
AVERAGE_REWARD_NEEDED_TO_END = 500
BATCH_SIZE = 40

GROUNDED_DISTANCE_THRESHOLD = 0.1 # The highest distance above a block for which the agent is considered to be stepping on it.

# training parameters
EPSILON = 0.5
GAMMA = 0.99
ALPHA = 0.7 # = [0,1], How much prioritization is used, with 0 being no prioritization
BETA = 0.5 # = [0,1], Annealing factor (I don't know what this is)
optimizer = keras.optimizers.Adam(learning_rate=0.0025, clipnorm=1.0)
loss_function = keras.losses.MeanSquaredError()


# states (state space is nearly infinite; not directly defined in our code)
# actions
actionNamesToActionsMap: dict() = {
    "stopMove": "move 0.0",
    "moveHalf": "move 0.5",
    "moveFull": "move 1.0",
    "jumpFull": "jump 1",
    "stopJump": "jump 0",
    "turnRight": "turn 1.0",
    "turnLeft": "turn -1.0",
    "stopTurn": "turn 0.0"
}
actionNames: list() = list(actionNamesToActionsMap.keys())
NUM_ACTIONS: int = len(actionNames)

# rewards
rewardsMap: dict() = {
    "steppedOnPreviouslySeenBlock": -5, # -0.2,
    "newBlockSteppedOn": 1000,
    "death": -50.0,
    "goalReached": 50000
}


# replay values
past_states = []
past_next_states = []
past_actions = []
past_rewards = []
past_priorities = []
importance_sampling_weight = []
episode_reward = 0
frame_number = 0
episode_number = 0




# models
def create_model():
    input_shape = (10,) # 3 for closest block, 3 for velocity vector, 2 for two boolean inputs

    inputs = layers.Input(shape=input_shape)

    layer1 = layers.Dense(16, activation='relu', name="layer1")(inputs)
    layer2 = layers.Dense(16, activation='relu', name="layer2")(layer1)
    layer3 = layers.Dense(16, activation='relu', name="layer3")(layer2)

    action = layers.Dense(NUM_ACTIONS, activation='linear')(layer3)

    return keras.Model(inputs=inputs, outputs=action)

model = create_model()
target_model = create_model()





# loss_func = tf.keras.losses.MSE()

# global agent variables
prev_agent_position = Vector(0.5, 227.0, 0.5) # Where the player was last update
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
    grid = observations.get(u'floor5x5x2')  
    player_location = [int(observations[u'XPos']), int(observations[u'YPos']), int(observations[u'ZPos'])]
    result = []
    # TODO: Make these variables
    i = 0
    for y in range(-1, 0 + 1):
        for z in range(-2, 2 + 1):
            for x in range(-2, 2 + 1):
                if grid[i] != "air" and grid[i] != "lava":
                    result.append(Block(player_location[0]+x, player_location[1]+y, player_location[2]+z, grid[i]))
                i += 1
    return result


def get_block_below_agent(observations):
    """
    Get an object representing the block beneath the agent's position.
    Does not count the block that the agent is occupying right now.

    returns: Block
    """
    block_name = observations.get(u'block_below_agent')[0]
    player_location = [int(observations[u'XPos']), int(observations[u'YPos']), int(observations[u'ZPos'])]
    return Block(player_location[0], player_location[1]-1, player_location[2], block_name)


def is_grounded(observations):
    """
    returns: bool: true if touching ground
    """
    grid = observations.get(u'floor5x5x2')  
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
    # Can check if observation doesn't contain necessary data.
    # if not u'XPos' in obs or not u'YPos' in obs or not u'ZPos' in obs:
    #     print("Does not exist")
    #     # TODO: Make something appropriate for when we are unable to get the agent position.
    #     return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)  
    # else:
    #     current_s = "%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
    #     print("Position: %s (x = %.2f, y = %.2f, z = %.2f)" % (current_s, float(obs[u'XPos']), float(obs[u'YPos']), float(obs[u'ZPos'])))
    #     print("Direction vector: (x = %.2f, y = %.2f, z = %.2f" % (float(obs[u'entities'][0][u'motionX']), float(obs[u'entities'][0][u'motionY']), float(obs[u'entities'][0][u'motionZ'])))

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

    # Velocity vector
    velocity = agent_position - prev_agent_position
    prev_agent_position = agent_position

    # Facing direction. Doesn't need to look up or down
    yaw = obs[u'Yaw']

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
        grounded_this_update
    )
    

# Don't think this is necessary because training loop function just loops episode as well
# def episode_loop():
#     """
#     Until the episode is done, repeat the same
#     (state1, action1, result1, state2, action2, ...)
#     steps in a loop
#     """
#     pass


def choose_action(model, state):
    """
    Called once per frame, to determine the next action to take given the current state
    Uses the value of epsilon to determine whether to choose a random action or the best action (via tf.argmax)
    
    if we want: update epsilon to decay toward its minimum
    """
    # return np.random.choice([actionNames.index("moveFull"), actionNames.index("jumpFull")])
    # return np.random.choice([actionNames.index("moveFull"), actionNames.index("jumpFull"), actionNames.index("turnLeft"), actionNames.index("stopTurn")])
    
    if np.random.rand(1)[0] < EPSILON:
        return np.random.choice(NUM_ACTIONS)
    else:
        action_probs = model(tf.expand_dims(tf.convert_to_tensor(state), 0), training=False)
        action = tf.argmax(action_probs[0]).numpy()
        return action


def obs_is_valid(raw_state):
    obs_text = raw_state.observations[-1].text
    obs = json.loads(obs_text)
    if not u'XPos' in obs or not u'YPos' in obs or not u'ZPos' in obs:
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
    
    returns: int, bool, integer reward and whether episode is done or not
    """
    reward = 0
    episode_done = False

    global blocks_walked_on
    global prev_block_below_agent
    world_state = raw_state

    obs_text = world_state.observations[-1].text
    obs = json.loads(obs_text) # most recent observation

    # need to update, but very rudimentary reward checking system
    # check for game finished
    grid = obs.get(u'floor5x5x2')  
    player_height = float(obs[u'YPos'])

    player_height_rounded = int(player_height)
    block_name_below_player = grid[5 * int(5 / 2) + int(5 / 2)] # TODO: Make this use variables or something
    if abs(player_height - player_height_rounded) <= 0.01:
        if(block_name_below_player == "diamond_block"):
            return rewardsMap["goalReached"], True
    
    agent_position_int = Vector(int(obs[u'XPos']), int(obs[u'YPos']), int(obs[u'ZPos']))
        
    grounded_this_update = is_grounded(obs)

    full_life = 20.0
    if obs.get(u"Life") < full_life:
        return 0, False

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
        return 0, False

    block_stepping_on = get_block_below_agent(obs)
    if (grounded_between_updates and prev_block_below_agent.name != "air" and prev_block_below_agent.name != "lava" 
        and prev_block_below_agent not in blocks_walked_on):
        # We were on a block between this update and the last. We only recorded that block from the previous update,
        # so use the stored value when adding which blocks we've stepped on.
        print("Stepped on a new block:", prev_block_below_agent.name, prev_block_below_agent.position(), "last update")
        blocks_walked_on.add(prev_block_below_agent)
        reward += rewardsMap["newBlockSteppedOn"]
    if block_stepping_on.name != "air" and block_stepping_on.name != "lava":
        # We have found the block we're stepping on, if any.
        # See if we have stepped on it before.
        if block_stepping_on.name != "stone":
            reward += 0
            episode_done = False
        elif block_stepping_on in blocks_walked_on:
            reward += rewardsMap["steppedOnPreviouslySeenBlock"]
            episode_done = False
        else:
            # TODO: Testing print to see when Agent thinks it's on a new block
            print("\nStepped on a new block:", block_stepping_on.name, block_stepping_on.position())
            blocks_walked_on.add(block_stepping_on)
            reward += rewardsMap["newBlockSteppedOn"]
            episode_done = False

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
    return reward, episode_done


def update_target_model():
    """
    """
    target_model.set_weights(model.get_weights())


def add_entry_to_replay(state, next_state, action, reward, priority):
    """
    Called every time the AI takes an action
    Updates the replay buffers in place

    returns: void
    """
    past_states.append(state)
    past_next_states.append(next_state)
    past_actions.append(action)
    past_rewards.append(reward)
    past_priorities.append(priority)
    importance_sampling_weight.append(1.0)


def reset_replay():
    past_states = []
    past_next_states = []
    past_actions = []
    past_rewards = []


def remove_first_entry_in_replay():
    """
    This function will be called when our replay buffers are longer than MAX_HISTORY_LENGTH
    """
    del past_states[0]
    del past_next_states[0]
    del past_actions[0]
    del past_rewards[0]
    del past_priorities[0]
    del importance_sampling_weight[0]
    # maybe print something


def training_loop(agent_host):
    global past_states
    global blocks_walked_on 
    blocks_walked_on.clear()
    episode_reward = -rewardsMap["newBlockSteppedOn"] # zeroes out the reward that's given for just stepping on the first block (which is automatic, has nothing to do with the model's choices)
    episode_done = False
    frame_number = 0
    episode_start_time = time.time()

    cur_state_raw = agent_host.getWorldState()
    while (len(cur_state_raw.observations) == 0) or (not obs_is_valid(cur_state_raw)):
        cur_state_raw = agent_host.getWorldState()
    cur_state = format_state(cur_state_raw)

    for _ in range(MAX_ACTIONS_PER_EPISODE):
        if len(past_next_states) > 0:
            cur_state = past_next_states[-1]
        action = choose_action(model, cur_state)
        take_action(action, agent_host)

        # time.sleep(0.05)

        goal_reached = False
        is_dead = False
        next_state_raw = agent_host.getWorldState()
        while (len(next_state_raw.observations) == 0) or (not obs_is_valid(next_state_raw)):
            next_state_raw = agent_host.getWorldState()
            if (not next_state_raw.is_mission_running):
                # TODO: Hack to check if player has reached the goal or not. There are multiple ways to end the mission
                if(next_state_raw.rewards and next_state_raw.rewards[-1].getValue() == rewardsMap["goalReached"]):
                # if cur_state[7]: # ISSUE: if player is jumping onto the diamond block, the mission ends but the reward is not given. it's only given via this condition if the agent walks onto the goal block
                    goal_reached = True
                else:
                    # Agent has died or fallen off the map
                    is_dead = True
                break
        frame_number += 1
        
        if is_dead:
            episode_time_taken = time.time() - episode_start_time
            next_state = cur_state
            reward, episode_done = rewardsMap["death"] / episode_time_taken, True
        elif not goal_reached:
            next_state = format_state(next_state_raw)
            reward, episode_done = calculate_reward(next_state_raw)

            # Remember previous block. Used in calculate_reward
            global prev_block_below_agent
            prev_block_below_agent = get_block_below_agent(json.loads(next_state_raw.observations[-1].text))
        else:
            episode_time_taken = time.time() - episode_start_time
            next_state = cur_state
            reward, episode_done = rewardsMap["goalReached"] / episode_time_taken, True
        episode_reward += reward

        # Calculate priority for prio replay
        if len(past_priorities) <= 1:
            priority = 1.0
        else:
            priority = max(past_priorities)

        add_entry_to_replay(cur_state, next_state, action, reward, priority)

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
                max_isw = max(importance_sampling_weight)
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
        
        """
        if frame_number % UPDATE_MODEL_AFTER_N_FRAMES == 0 and frame_number > BATCH_SIZE:
            # Choose past actions from a distribution proportional to their priority.
            # TODO: I'm pretty sure this isn't the correct distribution as described on the bottom of 
            # page 4 of the paper, under "Implementation". I don't know what it's actually asking for.
            p_alpha = np.power(past_priorities, ALPHA)
            p_alpha /= sum(p_alpha)
            random_indices = np.random.choice(range(len(past_states)), size=BATCH_SIZE, p=p_alpha)

            sampled_states = np.array([past_states[i] for i in random_indices])
            sampled_actions = np.array([past_actions[i] for i in random_indices])
            sampled_rewards = np.array([past_rewards[i] for i in random_indices])

            # Importance-sampling weight
            max_weight = max(importance_sampling_weight)
            for j in random_indices:
                importance_sampling_weight[j] = (MAX_HISTORY_LENGTH * p_alpha[j]) ** (-BETA) / max_weight
            
            # next_state = tf.convert_to_tensor(next_state)
            predicted_future_rewards = target_model.predict(sampled_states, verbose=0) # prints to stdout
            bellman_updated_q_vals = sampled_rewards + GAMMA * tf.reduce_max(predicted_future_rewards, axis=1)
            action_mask = tf.one_hot(sampled_actions, NUM_ACTIONS)

            with tf.GradientTape(persistent=True) as tape:
                original_q_vals = model(sampled_states)

                # Update priorities
                # TODO: Don't know if this is correct update (line 11-12 of paper pseudocode)
                target_prediction_values = target_model(sampled_states)
                td_error = []
                for rnd_i in range(len(random_indices)):
                    td_error.append(float(target_prediction_values[rnd_i, past_actions[rnd_i]] - original_q_vals[rnd_i, past_actions[rnd_i]]))
                    past_priorities[random_indices[rnd_i]] = abs(td_error[rnd_i])


                original_q_vals_for_actions = tf.reduce_sum(tf.multiply(original_q_vals, action_mask), axis=1)
                

                # loss = loss_function(bellman_updated_q_vals, original_q_vals_for_actions)
                # loss_function_returns.append(loss)
                
            # TODO:Testing gradient accumulation
            # gradients = []
            loss = 0.0
            for j in range(len(random_indices)):
                individual_loss = loss_function([float(bellman_updated_q_vals[j])], [float(original_q_vals_for_actions[j])])
                loss += individual_loss
                print(individual_loss)
                grads = tape.gradient(individual_loss, model.trainable_variables)
                print(grads)
                delta = importance_sampling_weight[random_indices[j]] * td_error[j] * grads
                if len(gradients) == 0:
                    gradients = delta
                else:
                    gradients += delta
            
            gradients = tape.gradient(loss, model.trainable_variables)
            print("num gradients:", gradients)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        """

        # if ready to update model (% UPDATE_MODEL_AFTER_N_FRAMES == 0)
        # take sample from replay buffers & update q-values
        # update value of episode_done
        if frame_number % UPDATE_TARGET_AFTER_N_FRAMES == 0:
            # if ready to update target model
            update_target_model()


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

            if goal_reached:
                episodes_that_succeeded.append(i)
            print("Episode reward:", episode_reward, "Average reward:", (reward_of_all_episodes / (i+1)), "Successful episodes:", episodes_that_succeeded)

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
    if reward_of_all_episodes / (i+1) > AVERAGE_REWARD_NEEDED_TO_END:
        print("AI too good")
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
    print("Mission running ", end=' ')

    # testing training loop function
    training_loop(agent_host)
    time.sleep(0.5)

    # decay epsilon (model should be more confident)
    EPSILON -= (EPSILON / NUM_EPISODES)
    print(EPSILON)






plt.title("Loss function return value over time")
plt.xlabel("# call to loss function")
plt.ylabel("Loss function return value (log scale)")
plt.semilogy(range(1, len(loss_function_returns) + 1), loss_function_returns, linewidth=2.0, c="red")
plt.show()

plt.title("Raw reward per episode over episode iterations")
plt.xlabel("Episode #")
plt.ylabel("Cumulative reward")
plt.plot(range(1, NUM_EPISODES+1), episode_rewards, linewidth=2.0, c="green")
plt.show()


plt.title("Average reward over episode iterations")
plt.xlabel("Episode #")
plt.ylabel("Average reward of current episode and past episodes")
plt.plot(range(1, NUM_EPISODES+1), episode_reward_running_avgs, linewidth=2.0, c="blue")
plt.show()


with open("./training_results.py", 'w') as f:
    f.write(str(episode_rewards) + "\n")
    f.write(str(episode_reward_running_avgs) + "\n")

tf.keras.utils.plot_model(target_model, to_file="model.png", show_shapes=True, rankdir="LR", dpi=300)
