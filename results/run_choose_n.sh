#!/bin/bash
# Which n to buy (batch only, per organiser ruling), full game, competition-day config adapt=1.
R=${R:-150}
S=("bayes:n=2,adapt=1" "bayes:n=3,adapt=1" "bayes:n=4,adapt=1" "bayes:n=5,adapt=1" "bayes:n=7,adapt=1")
mc() { python3 -m scavenger.montecarlo -r $R --workers 4 "$@" -s "${S[@]}" 2>/dev/null | grep "^bayes"; }
echo "## TIMID 0-50%";        mc
echo "## MIXED";              mc --bot-mix "0.4:0:0.15,0.4:0.1:0.3,0.2:0.6:1.0"
echo "## AGGRESSIVE 30-120%"; mc --bot-lo 0.3 --bot-hi 1.2
echo "## MIDDLING 20-80%";    mc --bot-lo 0.2 --bot-hi 0.8
echo "## FAIR SHARE 5-25%";   mc --bot-lo 0.05 --bot-hi 0.25
