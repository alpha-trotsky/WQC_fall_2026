#!/bin/bash
# Each block: the simulated game resolves one open rule a certain way; compare a strategy
# that assumes our default vs one that assumes the true rule. Same rounds for both.
R=${R:-300}
run() { echo "### truth: $1"; python3 -m scavenger.montecarlo -r $R $2 -s "${@:3}" | sed -n '/^strategy/,$p'; echo; }
run "two-point percentile = cdf (H always shows 100)" "--truth-tp cdf" "bayes:n=3" "bayes:n=3,tp=cdf"
run "two-point percentile = below (L always shows 0)" "--truth-tp below" "bayes:n=3" "bayes:n=3,tp=below"
run "samples are real-valued, not whole numbers" "--truth-delta 1e-6" "bayes:n=3" "bayes:n=3,delta=1e-6"
run "negative pools allowed" "--truth-pos none" "bayes:n=3" "bayes:n=3,pos=none"
run "auction rank counts from the highest bid" "--truth-rank high" "bayes:n=3" "bayes:n=3,rank=high"
