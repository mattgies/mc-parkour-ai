import tensorflow as tf
import numpy as np
from tensorflow import keras

"""
num_ticks = 0

MAIN LOOP:

ws = agent.getWorldState() OR past_states[-1]
# location vector
# direction vector
# isGrounded
# yaw
cur_state = tf.constant(list(loc) + list(dir) + list(float(isGrounded)) + list(float(yaw)))
past_states.append(cur_state)

choose an action
action = choose_action(cur_state) # epsilon-greedy, or argmax for action with the best q-value in the model's outputs

take action

# IMPORTANT
time.sleep(0.001) # IMPORTANT wait for the action to propogate and result in a new state
# this differs from the atari breakout scenario, because in that case the results were immediate
num_ticks += 1

new_state = getFormattedState()
past_states.append(new_state)

if num_ticks % 4 == 0:
    # do the updates

    # s = cur_state
    # s' = new_state
    # gamma = training constant
    # a = action
    # R(s') = reward from doing action a from state s; calculated as a function of the new_state
    # BELLMAN EQUATION: Q(s,a) = R(s') + gamma * max_a(s', a)

    predicted_future_rewards = target_model.predict(new_state)
    bellman_updated_q_vals = reward(new_state) + GAMMA * tf.reduce_max(predicted_future_rewardsm, axis=1)
    # gradient tape here
        original_q_vals = model(sampled_states)
        original_q_vals_for_actions = tf.reduce_sum(tf.multiply(original_q_vals, mask), axis=1)
        loss = loss_function(bellman_updated_q_vals, original_q_vals_for_actions)
    # backpropagation here
            
if num_ticks % 1000 == 0:
    update_target_model()

if len(state_history) > MAX_REPLAY_LENGTH:
    remove_first_entry_in_replay()

if episode_done(new_state):
    # mission finished, no need to continue taking actions
    break

"""






BATCH_SIZE = 50





rewardsMap: dict(str, float) = {
    "steppedOnPreviouslySeenBlock": -1
    "newBlockSteppedOn": 2,
    "death": -500.0,
    "goalReached": 500
}


actionNamesToActionsMap: dict(str, str) = {
    "stopMove": "move 0.0",
    "moveHalf": "move 0.5",
    "moveFull": "move 1.0",
    "jumpFull": "jump 1.0",
    "stopJump": "jump 0.0",
    "turnRight": "turn 1.0",
    "turnLeft": "turn -1.0",
    "stopTurn": "turn 0.0"
}
actionNames: list(str) = [action for action in actionNamesToActionsMap]
NUM_ACTIONS: int = len(actionNames)
if NUM_ACTIONS != len(actionNamesToActionsMap):
    raise IndexError("1+ actions are missing from actionNames or the actionNamesToActionsMap")

past_actions: list(int) = [] # each int corresponds to an index in the actions array

def save_values_into_replay(state, action, reward):
    past_states.append(state)
    past_actions.append(action)
    past_rewards.append(reward)
    pass


def update_model():
    # take random sample from replay buffers
    random_indices_for_sampling = np.random.choice(range(len(past_states)), size=BATCH_SIZE)
    sampled_states = [past_states[i] for i in random_indices_for_sampling]
    sampled_actions = [past_actions[i] for i in random_indices_for_sampling]
    sampled_rewards = [past_rewards[i] for i in random_indices_for_sampling]
    
    predicted_future_rewards = target_model.predict(next_state)
    bellman_new_q_vals = sampled_rewards + GAMMA * tf.reduce_max(predicted_future_rewards, axis=1)

    mask = tf.one_hot(sampled_actions, NUM_ACTIONS)
    
    with tf.GradientTape() as tape:
        q_vals = model(sampled_states)
        q_action = tf.reduce_sum(tf.multiply(q_vals, mask), axis=1)
        loss = loss_function(bellman_new_q_vals, q_action)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradientrs(zip(gradients, model.trainable_variables))


def update_target_model(target_model: keras.Model, model: keras.Model):
    """
    """
    target_model.set_weights(model.get_weights())
    pass
