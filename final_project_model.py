import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# CONSTANTS
MAX_HISTORY_LENGTH = 50000
MAX_ACTIONS_PER_EPISODE = 10000
UPDATE_MODEL_AFTER_N_FRAMES = 5
UPDATE_TARGET_AFTER_N_FRAMES = 5000

# training parameters
epsilon = 0.2

# rewards
{
    "death": -5000,
    "goingBackwards": -2,
    "goingForwards": 2,
    "reachedGoal": 5000
}


# replay values
past_states = []
past_actions = []
past_rewards = []
episode_reward = 0
frame_number = 0
episode_number = 0

num_actions = 5 # for now actions are (move (forward), turn left, turn right, stop, jump)

def create_model():
    input_shape = (8, ) # 3 for closest block, 3 for velocity vector, 2 for two boolean inputs

    inputs = layers.Input(shape=input_shape)

    layer1 = layers.Flatten()(inputs)
    layer2 = layers.Dense(16, activation='relu')(layer1)
    layer3 = layers.Dense(16, activation='relu')(layer2)
    layer4 = layers.Dense(16, activation='relu')(layer3)

    action = layers.Dense(num_actions, activation='linear')(layer4)

    return keras.Model(inputs=inputs, output=action)


def get_init_state():
    """
    Called once, returns the start state observations

    Returns 3 values, a tuple length of 3 of the closest block, a tuple length 3 of the velocity vector, a
    float for look direction, a boolean for isGrounded
    """
    # For now, the observation is on the other file, so we need to change the structure somehow
    return (0, 0, 0, 0, 0, 0, 0, 1)


def get_next_state(state, action):
    """
    Called once a frame, returns the next state after action is performed
    
    Returns 3 values, a tuple length of 3 of the closest block, a tuple length 3 of the velocity vector, a
    float for look direction, a boolean for isGrounded
    """
    # For now, the observation is on the other file, so we need to change the structure somehow
    return (0, 0, 0, 0, 0, 0, 0, 1)


def choose_action(ep, model, state):
    """
    Called once per frame, to determine the next action to take given the current state
    Uses the value of epsilon to determine whether to choose a random action or the best action (via tf.argmax)
    
    if we want: update epsilon to decay toward its minimum
    """
    if np.random.rand(1)[0] < ep:
        return np.random.choice(num_actions)
    else:
        action_probs = model(tf.convert_to_tensor(state), training=False)
        action = tf.argmax(action_probs[0]).numpy()
        return action


# def episode_loop():
#     """
#     Until the episode is done, repeat the same
#     (state1, action1, result1, state2, action2, ...)
#     steps in a loop
#     """
#     pass



def take_action(action):
    """
    Called once per frame, after choose_action
    
    returns: next state, float, representing Reward, boolean if on finish block
    """
    pass


def update_target_model():
    """
    """
    pass


def add_entry_to_replay(state, action, reward):
    """
    Called every time the AI takes an action
    Updates the replay buffers in place

    returns: void
    """
    past_states.append(state)
    past_actions.append(action)
    past_rewards.append(reward)


def remove_first_entry_in_replay():
    """
    This function will be called when our replay buffers are longer than MAX_HISTORY_LENGTH
    """
    del past_states[0]
    del past_actions[0]
    del past_rewards[0]
    # maybe print something


def training_loop():
    model = create_model()
    target_model = create_model()

    optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

    state = get_init_state()

    while True:
        episode_reward = 0
        episode_done = False

        for _ in range(MAX_ACTIONS_PER_EPISODE):
            frame_number += 1

            action = choose_action(epsilon, model, state)
            next_state, reward, episode_done += take_action(action)

            episode_reward += reward
            
            add_entry_to_replay(state, action, reward)

            # if ready to update model (% UPDATE_MODEL_AFTER_N_FRAMES == 0)
            # take sample from replay buffers & update q-values
            # update value of episode_done
            if frame_number % UPDATE_MODEL_AFTER_N_FRAMES == 0:
                # if ready to update target model
                update_target_model()


            if len(past_states) > MAX_HISTORY_LENGTH:
                remove_first_entry_in_replay()

            if episode_done:
                break
