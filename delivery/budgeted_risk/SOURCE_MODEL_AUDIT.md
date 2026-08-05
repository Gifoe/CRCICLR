# Source-model audit

The inherited selected heads were rejected because their training subjects overlap Stage-0 calibration/evaluation and reserved internal subjects. No uncontaminated inherited head was found. Clean cross-fitted copies of the existing linear/task-head architectures were therefore trained only on the three meta folds for every outer fold and seed; frozen CBraMod token embeddings were used and the backbone was not updated.

- clean heads: 50
- evaluation overlap: 0
- calibration overlap: 0
- formal overlap: 0
- internal-final overlap: 0
- CAP overlap: 0
- aggregate source hash: `043b21dd78a76c59252ad20485e1f3fc4398ddcf0ebddb786c8a1f62b70d4901`
