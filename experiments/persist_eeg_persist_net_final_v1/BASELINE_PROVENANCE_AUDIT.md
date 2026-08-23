# Baseline provenance audit

## Historical evidence

- The old 83.775% checkpoint is impure for the present V8 split and its weights
  are forbidden.
- The latest clean strong generic audit reported 77.275% on 40 V8_SEARCH
  subjects.  It is not accepted as the automatic comparison ceiling.
- The historical MI-specific training recipe (standard EEGNet candidates,
  source-subject validation, legal Session-1 target adaptation) achieved
  83.204% on its older 54-subject protocol.  Its recipe is reusable, but its
  learned weights and old outcome-based selections are not.
- V6/V7 recipes are provenance evidence only.  All weights used here are
  freshly trained inside each current outer development fold.

## Reconstructed baselines

- **B0**: vanilla EEGNet F1=8, F2=16, embedding=64, task-only.
- **B1**: strongest of the two predeclared legal EEGNet widths (8/16 and
  16/32), selected using only inner source subjects and then refit.
- **B2**: B1 plus the strongest predeclared nonzero target-history adaptation
  recipe selected on inner source S1-to-S2 episodes.
- **B3/A2**: parameter-nearest dual-path EEGNet, task-only, with the same
  protected-branch freeze and adaptive-branch update budget as FULL.

The strongest comparable baseline for G1 is the highest legal mean BA among
B0, B1, and B2 on the identical folds/seeds.  The code does not hard-code
77.275% as the reference.
