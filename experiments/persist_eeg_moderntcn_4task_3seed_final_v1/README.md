# ModernTCN four-task three-seed matched baseline

This experiment applies the official ModernTCN classification architecture to the frozen CRCICLR subject-independent, future-session protocol. Runtime caches, checkpoints, logs, and trial prediction arrays are external to Git.

The committed protocol freeze must hash-match before a training cell can start. Outer-development training contains no heldout loader. Heldout evaluation is a separate post-freeze stage after all 60 selected checkpoints exist.

