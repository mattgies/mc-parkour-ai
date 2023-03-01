from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object
import numpy as np
import math
import MalmoPython
import tensorflow as tf
import json
import copy
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

# Classes
class Block:
    """
    Stores information about a Minecraft block.
    """
    def __init__(self, x, y, z, name):
        self.x = x
        self.y = y
        self.z = z
        self.name = name

    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + "," + str(self.z) + "|" + self.name + ")"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z
    
    def position(self) -> "Vector":
        return Vector(self.x, self.y, self.z)
    

class Vector:
    """
    Stores a 3D vector.
    """
    def __init__(self, x, y, z) -> None:
        self.x = x
        self.y = y
        self.z = z

    def __str__(self) -> str:
        return "(" + str(self.x) + "," + str(self.y) + "," + str(self.z) + ")"

    def __sub__(self, other) -> "Vector":
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y and self.z == other.z
    
    def magnitude(self) -> float:
        return math.sqrt((self.x ** 2) + (self.y ** 2) + (self.z ** 2))
    
    def direction(self) -> "Vector":
        mag = self.magnitude()
        return Vector(self.x / mag, self.y / mag, self.z / mag)


# functions
def GetMissionXML(summary=""):
    ''' Build an XML mission string. '''

    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
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
                    <DrawBlock x="2" y="226" z="2" type="diamond_block"/>
                </DrawingDecorator>
                <ServerQuitWhenAnyAgentFinishes />
            </ServerHandlers>
        </ServerSection>

        <AgentSection mode="Survival">
            <Name>ParkourPeter</Name>
            <AgentStart>
                <Placement x="0.5" y="227.0" z="0.5"/>
            </AgentStart>
            <AgentHandlers>
                <ContinuousMovementCommands turnSpeedDegs="480"/>
                <AbsoluteMovementCommands/>
                <SimpleCraftCommands/>
                <MissionQuitCommands/>
                <InventoryCommands/>
                <ObservationFromFullStats/>
                <ObservationFromGrid>
                    <!-- Observe blocks that are below and at leg-level of the agent. -->
                    <Grid name="floor5x5x2">
                        <min x="-2" y="-1" z="-2"/>
                        <max x="2" y="0" z="2"/>
                    </Grid>
                </ObservationFromGrid>
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


def is_grounded(observations, nearby_blocks):
    """
    returns: bool: true if touching ground
    """
    # TODO: Make own "IsClose" function for float math stuff
    grid = observations.get(u'floor5x5x2')  
    player_height = float(observations[u'YPos'])
    player_height_rounded = int(player_height)
    block_name_below_player = grid[5 * int(5 / 2) + int(5 / 2)] # TODO: Make this use variables or something
    return block_name_below_player != "lava" and block_name_below_player != "air" and (abs(player_height - player_height_rounded) <= 0.01)
    

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

time.sleep(1)

# Simulate running for a few seconds
prev_agent_position = Vector(0.5, 227.0, 0.5) # Where the player was last update

agent_host.sendCommand("move 2")
agent_host.sendCommand("jump 1")
agent_host.sendCommand("turn 1")
for update_num in range(80):
    print("Update num:", update_num)

    # TODO: Could do try/catch and skip this loop if it gives an out of bounds exception.
    # NOTE: Getting the observations multiple times on the same frame most likely causes the out of bounds exception.
    # Get agent observations for this update.
    world_state = agent_host.getWorldState()
    obs_text = world_state.observations[-1].text
    obs = json.loads(obs_text) # most recent observation
    # Can check if observation doesn't contain necessary data.
    if not u'XPos' in obs or not u'ZPos' in obs:
        print("Does not exist")
    # else:
    #     current_s = "%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
    #     print("Position: %s (x = %.2f, y = %.2f, z = %.2f)" % (current_s, float(obs[u'XPos']), float(obs[u'YPos']), float(obs[u'ZPos'])))
    #     print("Direction vector: (x = %.2f, y = %.2f, z = %.2f" % (float(obs[u'entities'][0][u'motionX']), float(obs[u'entities'][0][u'motionY']), float(obs[u'entities'][0][u'motionZ'])))

    # Where agent is this update.
    agent_position = Vector(float(obs[u'XPos']), float(obs[u'YPos']), float(obs[u'ZPos']))

    # Get grid observations
    blocks = get_nearby_walkable_blocks(obs)
    for b in blocks:
        if b.name == "diamond_block":
            direction_vector = b.position() - agent_position
            print("Magnitude:", direction_vector.magnitude(), "| Direction:", direction_vector.direction())

    # Grounded check
    print("Is grounded:", is_grounded(obs, blocks))

    # Velocity vector
    print("Velocity:", agent_position - prev_agent_position)
    prev_agent_position = agent_position

    # Facing direction. Doesn't need to look up or down
    print("Look direction:", obs[u'Yaw'])

    time.sleep(0.05)

time.sleep(1)

