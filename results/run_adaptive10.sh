#!/bin/bash
# Adaptive strategy with 10 candidate rooms, n=3, in 6 rooms. The last two are off the candidate grid.
R=${R:-200}
mc() { python3 -m scavenger.montecarlo -r $R "$@" -s "bayes:n=3,adapt=1" | grep "^bayes"; }
echo "## n=3, adapt=1 (10 candidate rooms)"
echo "TIMID 0-50%:";              mc
echo "MIXED:";                    mc --bot-mix "0.4:0:0.15,0.4:0.1:0.3,0.2:0.6:1.0"
echo "AGGRESSIVE 30-120%:";       mc --bot-lo 0.3 --bot-hi 1.2
echo "MIDDLING 20-80%:";          mc --bot-lo 0.2 --bot-hi 0.8
echo "OFF-GRID 10-60%:";          mc --bot-lo 0.1 --bot-hi 0.6
echo "OFF-GRID fair share 5-25%:"; mc --bot-lo 0.05 --bot-hi 0.25
