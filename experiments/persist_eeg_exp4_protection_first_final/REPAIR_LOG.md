# Repair log

* `7735c37c`: fixed the redundant anchor forward pass, fail-closed outer status, and Windows DataLoader worker configuration before V1 execution.
* `7ca8c26c`: added the exact decision-response-preserving V2 transform after observing V1 mechanism drift.
* `68d2a788`: added a fixed-strength soft response correction for V3 after V2's documented over-constraint.

All three executions have separate compact output roots. No historical result was silently overwritten and no outer data were accessed during repair or iteration.
