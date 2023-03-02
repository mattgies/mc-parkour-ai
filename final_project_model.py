import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

num_actions = 5 # for now actions are (move (forward), turn left, turn right, stop, jump)

def create_model():
    input_shape = (3, 3, 2) # 3 for closest block, 3 for velocity vector, 2 for two boolean inputs

    inputs = layers.Input(shape=input_shape)

    layer1 = layers.Flatten()(inputs)
    layer2 = layers.Dense(16, activation='relu')(layer1)
    layer3 = layers.Dense(16, activation='relu')(layer2)
    layer4 = layers.Dense(16, activation='relu')(layer3)

    action = layers.Dense(num_actions, activation='linear')(layer4)

    return keras.Model(inputs=inputs, output=action)


def choose_action(ep, model, input):
    """
    Called once per frame, to determine the next action to take given the current state
    Uses the value of epsilon to determine whether to choose a random action or the best action (via tf.argmax)
    
    if we want: update epsilon to decay toward its minimum
    """
    if np.random.rand(1)[0] < ep:
        return np.random.choice(num_actions)
    else:
        action_probs = model(tf.convert_to_tensor(input), training=False)
        action = tf.argmax(action_probs[0]).numpy()
        return action


def update_target_model():
    """
    """
    pass
