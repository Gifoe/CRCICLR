# Historical method difference audit

## P4-SI

P4-SI used one shared EEGNet embedding, estimated persistent/nuisance
coordinates, and applied selective subject-adversarial pressure.  PERSIST-Net
uses two parameter-independent functional pathways.  Its core objective does
not use subject identity, and its protected output is frozen for every target.

## Protection-First

Protection-First learned an ordinary EEGNet and post-hoc projected an adapter
update away from selected coordinates.  PERSIST-Net distills the teacher's
source-certified decision contribution during source training.  The protected
object is a task function (centered logits), not a post-hoc coordinate set.

## Utility-Preservation V2

Utility-Preservation V2 retained one anchor plus a residual adapter and an
intervention-utility hinge.  Here the student architecture itself is split
into protected and adaptive raw-EEG pathways, with separate heads and no
target-trainable parameter capable of changing the protected output.

## Guard, router, and action selector experiments

Those experiments attempted `PUD -> predict future harm/action`.  The latest
audit found exact finite D no better than simple confidence/update controls
for prospective harm and insufficient action-oracle headroom.  PERSIST-Net
does not predict harm, route subjects, or choose an action.  Every target uses
the same structural freeze rule.

## PB / orthogonal-subspace methods

No hard orthogonality between persistent and task structure is assumed or
penalized.  Historical erasure evidence shows they can be entangled.  A
whitened source spectrum is used only to certify finite teacher-function
interventions; the two student pathways are not forced to be orthogonal.
