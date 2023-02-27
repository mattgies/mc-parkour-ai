import numpy as np
import MalmoPython
import tensorflow as tf
import os
import sys
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
def GetMissionXML(summary=""):
    ''' Build an XML mission string. '''

    return '''<?xml version="1.0" encoding="UTF-8" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <About>
            <Summary>Super cool parkour bot!</Summary>
        </About>

        <ModSettings>
            <MsPerTick>50</MsPerTick>
        </ModSettings>

        <ServerSection>
            <ServerInitialConditions>
                <Time>
                    <StartTime>6000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
                <AllowSpawning>false</AllowSpawning>
            </ServerInitialConditions>
            <ServerHandlers>
                <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1" />
                <DrawingDecorator>
                    <!--Draw shapes/blocks here. List of commands at https://microsoft.github.io/malmo/0.21.0/Schemas/MissionHandlers.html#element_DrawBlock -->
                    <DrawBlock x="5" y="226" z="5" type="diamond_block"/>
                </DrawingDecorator>
                <ServerQuitWhenAnyAgentFinishes />
            </ServerHandlers>
        </ServerSection>

        <AgentSection mode="Survival">
            <Name>ParkourPeter</Name>
            <AgentStart>
                <Placement x="0.5" y="227.0" z="0.5"/>
                <Inventory>
                    <InventoryItem slot="9" type="planks" variant="acacia"/>
                    <InventoryItem slot="10" type="brown_mushroom"/>
                    <InventoryItem slot="11" type="planks" variant="spruce"/>
                    <InventoryItem slot="12" type="brown_mushroom"/>
                </Inventory>
            </AgentStart>
            <AgentHandlers>
                <ContinuousMovementCommands turnSpeedDegs="480"/>
                <AbsoluteMovementCommands/>
                <SimpleCraftCommands/>
                <MissionQuitCommands/>
                <InventoryCommands/>
                <ObservationFromNearbyEntities>
                    <Range name="entities" xrange="40" yrange="40" zrange="40"/>
                </ObservationFromNearbyEntities>
                <ObservationFromFullInventory/>
                <AgentQuitFromCollectingItem>
                    <Item type="rabbit_stew" description="Supper's Up!!"/>
                </AgentQuitFromCollectingItem>
            </AgentHandlers>
        </AgentSection>

    </Mission>'''

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


# Create default Malmo objects:
expected_reward = 3390
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
my_mission_record = MalmoPython.MissionRecordSpec()

# Attempt to start a mission:
max_retries = 3
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

agent_host.sendCommand("move 2")
time.sleep(2)
world_state = agent_host.getWorldState()
import json
import copy
x = copy.deepcopy(world_state.observations[0].text)
print(json.loads(x)["entities"][0]["motionZ"])

time.sleep(5)

