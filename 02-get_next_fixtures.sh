#!/bin/bash

NUM_AHEAD=5

if [ "$#" -lt 1 ]
then
    echo "ERROR: No starting gameweek supplied."
    exit 1
fi

STARTING_GAMEWEEK=$1
MAX_GAMEWEEK=$((STARTING_GAMEWEEK+NUM_AHEAD-1))

for i in $(seq $STARTING_GAMEWEEK $MAX_GAMEWEEK)
do
    echo "Getting fixtures for gameweek $i..."
    curl "https://fantasy.premierleague.com/api/fixtures/?event=$i" -o "data/fixture_data/26/fixture_data_$i.json"
done

# Account for off-by-1 indexing errors
curl "https://fantasy.premierleague.com/api/event/${STARTING_GAMEWEEK}/live/" -o "data/raw/weekly_points/26/$((${STARTING_GAMEWEEK}-1)).json"

echo "All done!"