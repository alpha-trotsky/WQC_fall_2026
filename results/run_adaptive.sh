#!/bin/bash
# Adaptive room model vs fixed beliefs, in 4 rooms (the 4th is not one of the model's candidates).
R=${R:-200}
mc() { python3 -m scavenger.montecarlo -r $R "$@" | sed -n '/^strategy/,/^placement/p' | grep -v '^placement'; echo; }
S=("bayes:n=1,adapt=1" "bayes:n=3,adapt=1" "bayes:n=5,adapt=1")
echo "## TIMID room (0-50%)"; mc -s "${S[@]}"
echo "## MIXED room"; mc --bot-mix "0.4:0:0.15,0.4:0.1:0.3,0.2:0.6:1.0" -s "${S[@]}"
echo "## AGGRESSIVE room (30-120%)"; mc --bot-lo 0.3 --bot-hi 1.2 -s "${S[@]}"
echo "## MIDDLING room (20-80%, not a candidate)"; mc --bot-lo 0.2 --bot-hi 0.8 -s "${S[@]}" "bayes:n=3"
