import tensorflow as tf
import numpy as np
from tensorflow import keras

BATCH_SIZE = 50





rewardsMap: dict(str, float) = {
    "steppedOnPreviouslySeenBlock": -1
    "newBlockSteppedOn": 2,
    "death": -500.0,
    "goalReached": 500
}








actionNames: list(str) = []
actionNamesToActionsMap: dict(str, str) = {
    "actionName": "move 1.0"
    # add more
}
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
