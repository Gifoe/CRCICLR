# Leakage audit

The source runner reads only CleanRoom source archives. Historical adapters are
fit from the earliest session, split by sorted trial index. The later session
is passed only to metric functions. Recipe selection uses validation later-
session metrics but never later-session labels for fitting or Fisher pooling.
Every emitted row sets `future_session_used_for_fit=false` and
`future_labels_used_for_fit=false`. The fail-closed future lock rejects all
future access unless `status=AUTHORIZED` and `source_gate_pass=true`.
