# Resource admission incident

During the corrected EEGNet-MI fold0 attempt, an unrelated server process `resnet_3ds_server.py` used approximately 31 GB physical RAM. Free physical RAM fell to approximately 6 GB, and combined GPU usage reached approximately 31 GB. This is not a scientific result.

Only the arbitration fold0 task/process was stopped for resource protection. Other workloads were left running. Its runtime log and any completed mapping checkpoints are preserved. Restart is authorized as recovery from this resource interruption, without changing outcomes or protocol.

The supervisor admits at most two cells, only when at least 40 GB physical RAM and 16000 MiB GPU memory are free at launch. These launch checks are conservative admission rules, not a guarantee against later resource use by unrelated processes. The heartbeat must continue checking live memory and task state.
