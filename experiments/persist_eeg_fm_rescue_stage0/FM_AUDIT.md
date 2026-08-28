# Foundation-model audit

- CBraMod official repository commit `b9e961003214326972c567eff390e75b0287e32a`, checkpoint SHA-256 `0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`; strict load succeeds.
- LaBraM official repository commit `c431221e6cfd23dbfa9950e0180682fb322b0548`, checkpoint SHA-256 `7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c`; pretraining heads are intentionally omitted and the two-class downstream head is newly initialized.
- Both primary representations are the official final encoder representation and are 200-dimensional. No layer search is allowed.
- ST-EEGFormer-Small remains unopened unless the frozen trigger fires.
