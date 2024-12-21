NUM_AHEAD=21

STARTING_GAMEWEEK=$1
MAX_GAMEWEEK=$((STARTING_GAMEWEEK+NUM_AHEAD-1))

for i in $(seq $STARTING_GAMEWEEK $MAX_GAMEWEEK)
do
    echo "Getting fixtures for gameweek $i..."
    curl "https://fantasy.premierleague.com/api/fixtures/?event=$i" -o "data/fixture_data/fixture_data_$i.json"
done

echo "All done!"