# H1 XS representation-level residual bottleneck — seed 0

Exact LiteBN-XS checkpoint per task/fold; all XS parameters and the original classifier are frozen.
Adapter: z_new = z + Up(GELU(Down(z))), Down 128->16, Up 16->128.
Up.weight and Up.bias are exactly zero at step 0; Down uses the standard Linear initialization.
Only Down/Up are optimized on precomputed frozen XS embeddings.
Canonical inner_train_subjects / inner_val_subjects only; no repeated inner splits.
lr=3e-05; weight_decay=0.0005; beta=0.001; batch=512; clip=5.0; max_steps=128.
evaluation_steps=[0, 2, 4, 8, 16, 32, 64, 96, 128]; selection=highest canonical inner-val subject-mean BA, then smaller residual norm, then earlier step.
ERP clear-signal operational threshold=0.1 pp; smaller means are treated as essentially zero.
Frozen split sha256=d313f8e1f2c105d0b387c607fb781b61d9a05d92875759e51ed0d4c6d889f42a
H0 helper sha256=1b3e29ce83d59594bef439d46a89e6767297292423462923a6c2bf197efbe90d
FINAL_HELDOUT_ACCESSED = NO
