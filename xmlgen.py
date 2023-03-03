# multiple grids (with names) in ObsFromGrid
# min/max values per grid
# DrawingDecorator components (DrawCuboid)
from worldClasses import *

def cubeTags(cube_coords: list(Vector)):
    genstr = ""
    for cube in cube_coords:
        genstr += '<DrawBlock x="%d" y="%d" z="%d" type="grass"/>\n' % (cube)
    return genstr

def goalBlock(goal_coords: list(Vector)):
    return '<DrawBlock x="%d" y="%d" z="%d" type="diamond_block"/>\n' % (goal_coords)

def observationGrids(observation_grids: list(dict("name", "min", "max"))):
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

def XMLGenerator(cube_coords: list(Vector), observation_grids: list(Vector)):
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
                <FlatWorldGenerator generatorString="3;252*0;12;biome_1,village" />
                <DrawingDecorator> ''' + cubeTags(cube_coords) + goalBlock(goal_coords) + ''' </DrawingDecorator>
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
                <ObservationFromGrid> ''' + observationGrids(observation_grids) + ''' </ObservationFromGrid>
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

    return xmlstring