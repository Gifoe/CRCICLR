# Theory note

Under conditional error rates e for a run and rho for the remaining-run consensus, disagreement raises the posterior error probability when `P(disagree|wrong) / P(disagree|correct) > 1`. If an alternative action has positive conditional utility only in that region, a KEEP-outside/mixture-inside policy can dominate unconditional mixing. These are explicit assumptions, not claims that EEG errors are independent. A deterministic simulation is included in `code/cgrfuse.py` tests.
