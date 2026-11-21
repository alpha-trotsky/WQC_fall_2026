#!/bin/bash
# rank=auto must work whichever way the game counts; tp=cdf tested against a game using "below".
R=${R:-300}
for t in low high; do echo "### truth: rank counts from $t"; python3 -m scavenger.montecarlo -r $R --truth-rank $t -s "bayes:n=3,rank=$t" "bayes:n=3,rank=auto" | sed -n '/^strategy/,/^bayes:n=3,rank=auto/p'; echo; done
echo "### truth: two-point percentile = below"; python3 -m scavenger.montecarlo -r $R --truth-tp below -s "bayes:n=3,tp=cdf" | sed -n '/^strategy/,/^bayes/p'
