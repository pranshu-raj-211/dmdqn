## How to run

## Generating files for new network

Build a network in netedit or use the netgenerate command to build one for you. Save this in an appropriate directory.

```bash
netgenerate --grid --grid.x-number 3 --grid.y-number 3 --grid.attach-length 100 --grid.x-length 400 --grid.y-length 400 --tls.guess --default.lanenumber 3 --lefthand --output-file grid_3x3_lht.net.xml
```

Use netedit to make additional changes.
`netedit grid_3x3_lht.net.xml`

After this we need to generate trips, so that we can simulate traffic.
In the directory where the network file was saved:

`python /usr/share/sumo/tools/randomTrips.py -n grid.net.xml -o trips.trips.xml -e 2500 -p 0.6 --fringe-factor 5`

If you have specific vehicles defined, in a file vtypes.xml:
```xml
<vTypes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" minGap="2.5" maxSpeed="30" guiShape="passenger"/>
    <vType id="truck" accel="1.5" decel="4.5" sigma="0.5" length="10.0" minGap="3.0" maxSpeed="20" guiShape="truck"/>
    <vType id="motorcycle" accel="3.0" decel="5.0" sigma="0.5" length="2.0" minGap="1.0" maxSpeed="40" guiShape="motorcycle"/>
    <vType id="bus" accel="2.0" decel="4.0" sigma="0.5" length="12.0" minGap="3.0" maxSpeed="25" guiShape="bus"/>
</vTypes>
```

use 
`python /usr/share/sumo/tools/randomTrips.py -n grid_3x3_lht.net.xml -o trips.rou.xml --begin 0 --end 3600 --period 1`

vehicles will need to be added manually, I prefer using a script to randomize things.


Where 
- grid.net.xml is your network file
- /usr/share/sumo/ is the SUMO_HOME, env var in linux
- trips.trips.xml is the output file (-o flag)
- 2500 duration (timesteps)
- 0.6 defines density (time one vehicle is generated in, measured in seconds)


`duarouter -n grid_3x3.net.xml --route-files trips_p05.trips.xml -o grid_3x3_p05.rou.xml --ignore-errors`