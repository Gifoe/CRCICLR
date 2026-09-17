# Full-60 protocol correction

Source archival commit: `e9c689f7f4147b2419b9cbf122c9c0a9a6a34a54`.

The earlier seed-0 execution used a patience-8 termination rule. This follow-up restores the authoritative fixed 60-epoch training horizon while holding the archived cache, subject splits, episode manifests, normalizers, models, seeds, optimizer state, AMP scaler state and RNG state fixed.

Only the archived `checkpoint_latest.pt` states were resumed; no model was reinitialized and no early-stopped run was restarted from epoch 1.
