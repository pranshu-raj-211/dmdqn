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

Where 
- grid.net.xml is your network file
- /usr/share/sumo/ is the SUMO_HOME, env var in linux
- trips.trips.xml is the output file
- 2500 duration (timesteps)
- 0.6 defines density (closer to 1 -> lower density)


`duarouter -n grid_3x3.net.xml --route-files trips_p05.trips.xml -o grid_3x3_p05.rou.xml --ignore-errors`