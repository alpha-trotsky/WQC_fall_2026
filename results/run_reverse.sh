#!/bin/bash
# The other direction: the game uses our defaults, but the strategy assumes something else.
R=${R:-300}
echo "### truth: defaults (whole numbers, random two-point percentile, rank from lowest)"
python3 -m scavenger.montecarlo -r $R -s "bayes:n=3" "bayes:n=3,delta=1e-6" "bayes:n=3,tp=cdf" "bayes:n=3,tp=below" "bayes:n=3,rank=high" | sed -n '/^strategy/,$p'
