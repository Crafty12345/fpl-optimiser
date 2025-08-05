#!/bin/bash

if [ "$#" -ne 1 ]
then
    echo "ERROR: No date supplied."
    exit 1
fi

currentDate=$1

curl https://fantasy.premierleague.com/api/bootstrap-static/ -o "./data/raw/player_stats/fpl_stats_$currentDate.json"