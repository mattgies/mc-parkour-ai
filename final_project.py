import numpy as np
import MalmoPython
import tensorflow as tf
import time

x = tf.constant(4)
for i in range(5):
    print(x)




# CONSTANTS
MAX_HISTORY_LENGTH = 50000
MAX_ACTIONS_PER_EPISODE = 10000
UPDATE_MODEL_AFTER_N_FRAMES = 5
UPDATE_TARGET_AFTER_N_FRAMES = 5000

# training parameters
epsilon = 0.2


# states (state space is nearly infinite; not directly defined in our code)
# actions
actions = {"move", "jump", ""}
# OR
actions = {
    "move_slow": "move 0.2",
    "move_medium": "move 0.5",
    "move_fast": "move 1.0",
    "jump_low": "jump 0.2",
    "jump_medium": "jump 0.5",
    "jump_max": "jump 1.0"
    ''' more here '''
}

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


# loss_func = tf.keras.losses.MSE()


# functions
def create_model():
    pass


def episode_loop():
    """
    Until the episode is done, repeat the same
    (state1, action1, result1, state2, action2, ...)
    steps in a loop
    """
    pass


def choose_action():
    """
    Called once per frame, to determine the next action to take given the current state
    Uses the value of epsilon to determine whether to choose a random action or the best action (via tf.argmax)
    
    if we want: update epsilon to decay toward its minimum
    """



def take_action():
    """
    Called once per frame, after choose_action
    
    returns: float, representing Reward
    """



def update_target_model():
    """
    """
    pass


def add_entry_to_replay():
    """
    Called every time the AI takes an action
    Updates the replay buffers in place

    returns: void
    """
    pass



def remove_first_entry_in_replay():
    """
    This function will be called when our replay buffers are longer than MAX_HISTORY_LENGTH
    """
    del past_states[0]
    del past_actions[0]
    del past_rewards[0]
    # maybe print something


def training_loop():
    while True:
        episode_reward = 0
        episode_done = False

        for _ in range(MAX_ACTIONS_PER_EPISODE):
            choose_action()
            episode_reward += take_action()
            add_entry_to_replay()

            # if ready to update model (% UPDATE_MODEL_AFTER_N_FRAMES == 0)
            # take sample from replay buffers & update q-values
            # update value of episode_done


            # if ready to update target model
            update_target_model()


            if len(past_states) > MAX_HISTORY_LENGTH:
                remove_first_entry_in_replay()

            if episode_done:
                break


time.sleep(5)

