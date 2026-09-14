# SGN four-task three-seed matched baseline

This experiment applies the authors' official SGN/SwinGroupNet classifier to the frozen CRCICLR subject-independent, future-session protocol. Runtime caches, checkpoints, logs, and trial prediction arrays are external to Git.

The committed protocol freeze must hash-match before training. Heldout evaluation is a separate post-freeze stage after all 60 selected checkpoints exist.

