#!/bin/bash
# A) Do the (unpublished) parameter ranges matter?  Game uses one preset, model assumes another.
# B) What n and what bidding, in three kinds of room?
R=${R:-200}
mc() { python3 -m scavenger.montecarlo -r $R "$@" | sed -n '/^strategy/,/^placement/p' | grep -v '^placement'; echo; }
echo "## A1 game ranges = WIDE";    mc --truth-shapes wide   -s "bayes:n=1" "bayes:n=1,shapes=wide" "bayes:n=3" "bayes:n=3,shapes=wide"
echo "## A2 game ranges = NARROW";  mc --truth-shapes narrow -s "bayes:n=1" "bayes:n=1,shapes=narrow" "bayes:n=3" "bayes:n=3,shapes=narrow"
echo "## A3 game ranges = DEFAULT"; mc -s "bayes:n=1" "bayes:n=1,shapes=wide" "bayes:n=3" "bayes:n=3,shapes=wide"
echo "## A4 pools far bigger than the model's range (M 5k-200k)"; mc --truth-mlo 5000 --truth-mhi 200000 -s "bayes:n=3" "bayes:n=3,mlo=1,mhi=1000000"
S=("bayes:n=1" "bayes:n=3" "bayes:n=5" "bayes:n=3,opp_hi=1.0" "bayes:n=3,bid=frac,frac=0.1" "naive:n=3")
echo "## B1 PASSIVE room (bots bid 0-50% of their estimate)"; mc -s "${S[@]}"
echo "## B2 MIXED room (40% bid 0-15%, 40% bid 10-30%, 20% bid 60-100%)"; mc --bot-mix "0.4:0:0.15,0.4:0.1:0.3,0.2:0.6:1.0" -s "${S[@]}"
echo "## B3 AGGRESSIVE room (bots bid 30-120%)"; mc --bot-lo 0.3 --bot-hi 1.2 -s "${S[@]}"
