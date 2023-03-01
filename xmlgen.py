
# multiple grids (with names) in ObsFromGrid
# min/max values per grid
# DrawingDecorator components (DrawCuboid)

def XMLGenerator(cube_coords):
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
                <FlatWorldGenerator generatorString="3;7,3*1,3*3,2,78;12;biome_1,village" />
                <DrawingDecorator>
                    <DrawBlock x="0" y="224" z="0" type="grass"/>
                    <DrawBlock x="0" y="224" z="1" type="grass"/>
                    <DrawBlock x="0" y="224" z="2" type="grass"/>
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

    return xmlstring