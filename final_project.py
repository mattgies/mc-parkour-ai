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
import parkourcourse2 as course
import observationgrid1 as obsgrid
from worldClasses import *
# stepped_on_blocks = {Block()}

x = tf.constant(4)
for i in range(5):
    print(x)



# CONSTANTS
MAX_HISTORY_LENGTH = 50000
MAX_ACTIONS_PER_EPISODE = 10000
UPDATE_MODEL_AFTER_N_FRAMES = 5
UPDATE_TARGET_AFTER_N_FRAMES = 5000
NUM_EPISODES = 500
AVERAGE_REWARD_NEEDED_TO_END = 500

# training parameters
EPSILON = 0.2
GAMMA = 0.9
optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
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
actionNames: list() = [action for action in actionNamesToActionsMap]
NUM_ACTIONS: int = len(actionNames)
if NUM_ACTIONS != len(actionNamesToActionsMap):
    raise IndexError("1+ actions are missing from actionNames or the actionNamesToActionsMap")

# rewards
rewardsMap: dict() = {
    "steppedOnPreviouslySeenBlock": -0.2,
    "newBlockSteppedOn": 200,
    "death": -1000.0,
    "goalReached": 2500
}


# replay values
BATCH_SIZE = 40
past_states = []
past_actions = []
past_rewards = []
episode_reward = 0
frame_number = 0
episode_number = 0




# models
def create_model():
    input_shape = (8,) # 3 for closest block, 3 for velocity vector, 2 for two boolean inputs

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
blocks_walked_on = set()
## Used for testing prints
reward_of_all_episodes = 0
episodes_that_succeeded = []

# functions
def GetMissionXML(summary=""):
    return xmlgen.XMLGenerator(
        cube_coords=course.CUBE_COORDS,
        observation_grids=obsgrid.OBSERVATION_GRIDS,
        goal_coords=course.GOAL_COORDS
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


def is_grounded(observations):
    """
    returns: bool: true if touching ground
    """
    # TODO: Make own "IsClose" function for float math stuff
    grid = observations.get(u'floor5x5x2')  
    player_height = float(observations[u'YPos'])
    player_height_rounded = int(player_height)
    block_name_below_player = grid[5 * int(5 / 2) + int(5 / 2)] # TODO: Make this use variables or something
    return block_name_below_player != "lava" and block_name_below_player != "air" and (abs(player_height - player_height_rounded) <= 0.01)


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
    if not u'XPos' in obs or not u'YPos' in obs or not u'ZPos' in obs:
        print("Does not exist")
        # TODO: Make something appropriate for when we are unable to get the agent position.
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)  
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

    return (direction_to_closest_unwalked_block.x,
            direction_to_closest_unwalked_block.y,
            direction_to_closest_unwalked_block.z,
            velocity.x,
            velocity.y,
            velocity.z,
            yaw,
            grounded_this_update)
    

# Don't think this is necessary because training loop function just loops episode as well
# def episode_loop():
#     """
#     Until the episode is done, repeat the same
#     (state1, action1, result1, state2, action2, ...)
#     steps in a loop
#     """
#     pass


def choose_action(ep, model, state):
    """
    Called once per frame, to determine the next action to take given the current state
    Uses the value of epsilon to determine whether to choose a random action or the best action (via tf.argmax)
    
    if we want: update epsilon to decay toward its minimum
    """
    # return actionNames.index("moveFull")
    
    if np.random.rand(1)[0] < ep:
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

    global blocks_walked_on
    world_state = raw_state

    # try:
    obs_text = world_state.observations[-1].text
    # except IndexError as err:
    #     raise ValueError("Unable to get new agent state.")

    obs = json.loads(obs_text) # most recent observation
    # Can check if observation doesn't contain necessary data.

    # need to update, but very rudimentary reward checking system
    # check for game finished
    grid = obs.get(u'floor5x5x2')  
    player_height = float(obs[u'YPos'])

    if (player_height < 100):
        reward += rewardsMap["death"]
        return reward, True

    player_height_rounded = int(player_height)
    block_name_below_player = grid[5 * int(5 / 2) + int(5 / 2)] # TODO: Make this use variables or something
    if abs(player_height - player_height_rounded) <= 0.01:
        if(block_name_below_player == "diamond_block"):
            reward += rewardsMap["goalReached"]
            return reward, True
    
    agent_position_int = Vector(int(obs[u'XPos']), int(obs[u'YPos']), int(obs[u'ZPos']))
        
    grounded_this_update = is_grounded(obs)

    if (not grounded_this_update):
        return 0, False

    blocks = get_nearby_walkable_blocks(obs)
    onOldBlock = False
    for b in blocks:
        if grounded_this_update and agent_position_int == b.position() + Vector(0,1,0):
            # We have found the block we're stepping on, if any.
            # See if we have stepped on it before.
            if b in blocks_walked_on:
                onOldBlock = True
            else:
                blocks_walked_on.add(b)
            break
    if onOldBlock:
        reward += rewardsMap["steppedOnPreviouslySeenBlock"]
    else:
        reward += rewardsMap["newBlockSteppedOn"]
    
    return reward, False


def update_target_model():
    """
    """
    target_model.set_weights(model.get_weights())


def add_entry_to_replay(state, action, reward):
    """
    Called every time the AI takes an action
    Updates the replay buffers in place

    returns: void
    """
    past_states.append(state)
    past_actions.append(action)
    past_rewards.append(reward)


def reset_replay():
    past_states = []
    past_actions = []
    past_rewards = []


def remove_first_entry_in_replay():
    """
    This function will be called when our replay buffers are longer than MAX_HISTORY_LENGTH
    """
    del past_states[0]
    del past_actions[0]
    del past_rewards[0]
    # maybe print something


def training_loop(agent_host):
    reset_replay()
    
    global blocks_walked_on 
    blocks_walked_on.clear()
    episode_reward = 0
    episode_done = False
    frame_number = 0
    cur_state_raw = agent_host.getWorldState()
    while (len(cur_state_raw.observations) == 0) or (not obs_is_valid(cur_state_raw)):
        cur_state_raw = agent_host.getWorldState()
    cur_state = format_state(cur_state_raw)

    for _ in range(MAX_ACTIONS_PER_EPISODE):
        if len(past_states) > 0:
            cur_state = past_states[-1]
        action = choose_action(EPSILON, model, cur_state)
        take_action(action, agent_host)

        # time.sleep(0.05)

        goal_reached = False
        is_dead = False
        next_state_raw = agent_host.getWorldState()
        while (len(next_state_raw.observations) == 0) or (not obs_is_valid(next_state_raw)):
            next_state_raw = agent_host.getWorldState()
            if (not next_state_raw.is_mission_running):
                # TODO: Hack to check if player has reached the goal or not. There are multiple ways to end the mission
                if cur_state[7]:
                    print("mission stopped running")
                    print(next_state_raw.rewards[0].getValue())
                    goal_reached = True
                else:
                    print("Agent has died or fallen off the map")
                    is_dead = True
                break
        frame_number += 1
        
        if is_dead:
            next_state = cur_state
            reward, episode_done = rewardsMap["death"], True
        elif not goal_reached:
            next_state = format_state(next_state_raw)
            reward, episode_done = calculate_reward(next_state_raw)
        else:
            next_state = cur_state
            reward, episode_done = rewardsMap["goalReached"], True
        episode_reward += reward

        add_entry_to_replay(next_state, action, episode_reward)
        
        if frame_number % UPDATE_MODEL_AFTER_N_FRAMES == 0 and frame_number > BATCH_SIZE:
            random_indices = np.random.choice(range(len(past_states)), size=BATCH_SIZE)
            sampled_states = np.array([past_states[i] for i in random_indices])
            sampled_actions = np.array([past_actions[i] for i in random_indices])
            sampled_rewards = np.array([past_rewards[i] for i in random_indices])
            
            # next_state = tf.convert_to_tensor(next_state)
            predicted_future_rewards = target_model.predict(sampled_states) # prints to stdout
            bellman_updated_q_vals = sampled_rewards + GAMMA * tf.reduce_max(predicted_future_rewards, axis=1)
            action_mask = tf.one_hot(sampled_actions, NUM_ACTIONS)

            with tf.GradientTape() as tape:
                original_q_vals = model(sampled_states)
                original_q_vals_for_actions = tf.reduce_sum(tf.multiply(original_q_vals, action_mask), axis=1)
                loss = loss_function(bellman_updated_q_vals, original_q_vals_for_actions)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
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
