# Emergence audit

Emergence requires two consecutive checkpoints passing persistence, signed utility, repaired decision dependence, subject-positive-fraction, and Holm gates. The first passing checkpoint is `t*`; rank selection is smallest passing rank.

|   fold |   rank | emerged   | t_star   | next_epoch   | utility_t_star   | utility_next   | decision_t_star   | decision_next   |
|-------:|-------:|:----------|:---------|:-------------|:-----------------|:---------------|:------------------|:----------------|
|      0 |      1 | False     |          |              |                  |                |                   |                 |
|      0 |      2 | False     |          |              |                  |                |                   |                 |
|      0 |      4 | False     |          |              |                  |                |                   |                 |
|      1 |      1 | False     |          |              |                  |                |                   |                 |
|      1 |      2 | False     |          |              |                  |                |                   |                 |
|      1 |      4 | False     |          |              |                  |                |                   |                 |
|      2 |      1 | False     |          |              |                  |                |                   |                 |
|      2 |      2 | False     |          |              |                  |                |                   |                 |
|      2 |      4 | False     |          |              |                  |                |                   |                 |
|      3 |      1 | False     |          |              |                  |                |                   |                 |
|      3 |      2 | False     |          |              |                  |                |                   |                 |
|      3 |      4 | False     |          |              |                  |                |                   |                 |
|      4 |      1 | False     |          |              |                  |                |                   |                 |
|      4 |      2 | False     |          |              |                  |                |                   |                 |
|      4 |      4 | False     |          |              |                  |                |                   |                 |
