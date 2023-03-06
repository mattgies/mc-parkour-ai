# multiple grids (with names) in ObsFromGrid
# min/max values per grid
# DrawingDecorator components (DrawCuboid)
from worldClasses import *

lava_floor = '''<DrawCuboid x1="-20" y1="225" z1="-20" x2="20" y2="225" z2="20" type="lava" />\n'''

def cubeTags(cube_coords):
    genstr = ""
    for cube in cube_coords:
        genstr += '<DrawBlock x="%d" y="%d" z="%d" type="stone"/>\n' % (cube)
    return genstr

def goalBlock(goal_coords):
    return '<DrawBlock x="%d" y="%d" z="%d" type="diamond_block"/>\n' % (goal_coords.x, goal_coords.y, goal_coords.z)

def observationGrids(observation_grids):
    # grid format:
    # {"name": str.
    #   "min": Vector()
    #   "max": Vector()
    # }
    genstr = ""
    for grid in observation_grids:
        min, max = grid["min"], grid["max"]
        genstr += '<Grid name="%s">\n\t<min x="%d" y="%d" z="%d"/>\n\t<max x="%d" y="%d" z="%d"/>\n</Grid>\n' % (grid["name"], min.x, min.y, min.z, max.x, max.y, max.z)
    return genstr

def XMLGenerator(cube_coords, observation_grids, goal_coords, goal_reward):
    xmlstring = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
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
                <FlatWorldGenerator generatorString="3;256*0;12;biome_1,village" />
                <DrawingDecorator> ''' + lava_floor + cubeTags(cube_coords) + goalBlock(goal_coords) + ''' </DrawingDecorator>
                <ServerQuitWhenAnyAgentFinishes />
            </ServerHandlers>
        </ServerSection>

        <AgentSection mode="Survival">
            <Name>ParkourPeter</Name>
            <AgentStart>
                <Placement x="0.5" y="227.0" z="0.5" pitch="45" />
            </AgentStart>
            <AgentHandlers>
                <ContinuousMovementCommands turnSpeedDegs="480"/>
                <AbsoluteMovementCommands/>
                <SimpleCraftCommands/>
                <MissionQuitCommands/>
                <InventoryCommands/>
                <ObservationFromFullStats/>
                <ObservationFromGrid> ''' + observationGrids(observation_grids) + ''' </ObservationFromGrid>
                <ObservationFromNearbyEntities>
                    <Range name="entities" xrange="40" yrange="40" zrange="40"/>
                </ObservationFromNearbyEntities>
                <ObservationFromFullInventory/>
                <RewardForTouchingBlockType>
                    <Block reward="''' + str(goal_reward) + '''" type="diamond_block" behaviour="onceOnly"/>
                </RewardForTouchingBlockType>
                <AgentQuitFromTouchingBlockType>
                    <Block type="diamond_block" />
                </AgentQuitFromTouchingBlockType>
            </AgentHandlers>
        </AgentSection>

    </Mission>'''

    return xmlstring