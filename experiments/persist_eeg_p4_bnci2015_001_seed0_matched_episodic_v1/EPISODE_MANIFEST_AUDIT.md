# Episode-manifest audit

The direct final OpenBMI sampler was called after adapting only dataset metadata: `0A -> session 1` and `1B -> session 2`. Its `name='OpenBMI'` branch creates the specified S1 support/S2 future query construction.

| Fold | Inner train | Inner val | Outer | Epochs | Steps/epoch | Episodes | Support trials | Query trials | Disjoint episodes | Manifest SHA-256 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 8 | sub-02 | sub-03,sub-08,sub-10 | 60 | 20 | 1200 | 76800 | 76800 | 1200 | `0ff569e763676687614840f2f230351d6311099f2d871dc4c905d1bcc1b55356` |
| 1 | 8 | sub-11 | sub-05,sub-06,sub-12 | 60 | 20 | 1200 | 76800 | 76800 | 1200 | `467c3f2c9a30d554135d9fb30abddf58560e96b7ca8d55828c979bdb569f9f0b` |
| 2 | 9 | sub-09 | sub-01,sub-04 | 60 | 20 | 1200 | 76800 | 76800 | 1200 | `fa81cff625bcb5e67ef5b4c6c5838be5e40102e4326819f66b02172af31d47c1` |
| 3 | 9 | sub-03 | sub-07,sub-11 | 60 | 20 | 1200 | 76800 | 76800 | 1200 | `ab923dbee8cf272aaa95904932d884f7c84d8bfc511bee5d2d2d30078d8956b0` |
| 4 | 9 | sub-10 | sub-02,sub-09 | 60 | 20 | 1200 | 76800 | 76800 | 1200 | `697172d4d66f4490a4182146fd9e1256016c7f562246759b0387262441fb3e15` |

Every audit checks 4 support and 4 non-overlapping query subjects, 8 trials/class/subject, 64+64 trials, S1-only support, S2-only query, and exclusion of inner-val/outer subjects.
