# Method

PERSIST-Guard is a risk-gated interpolation from a clean population EEGNet head to a source-selected target-history Generic head. Protected directions are estimated in the same 64-dimensional EEGNet representation and must pass source-only cross-session persistence (P), signed utility loss under erasure (U), and decision-flip coupling (D). Identity is measured only as a control. Low-risk subjects use Generic exactly; high-risk subjects receive the smallest source-selected rollback/shrink action.
