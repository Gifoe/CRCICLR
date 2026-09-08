# Method

All models see the same cached MI epochs and frozen five-fold subject split. OpenBMI trains on S1 and evaluates S2; WBCIC trains on S1+S2 and evaluates S3. New backbones use fixed recipes recorded before outer-dev inspection. Primary fusion is the arithmetic mean of raw pre-softmax logits.
