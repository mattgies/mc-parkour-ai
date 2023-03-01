# multiple grids (with names) in ObsFromGrid
# min/max values per grid
# DrawingDecorator components (DrawCuboid)

def cubeTags(cube_coords):
    genstr = ""
    for cube in cube_coords:
        genstr += '<DrawBlock x="%d" y="%d" z="%d" type="grass"/>\n' % (cube)
    return genstr

def XMLGenerator(cube_coords, observation_grids):
    # grid format:
    # {"gridName": str.
    #   "gridMin": (x1, y1, z1)
    #   "gridMax": (x2, y2, z2)
    # }
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
                <DrawingDecorator> ''' + cubeTags(cube_coords) + ''' </DrawingDecorator>
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