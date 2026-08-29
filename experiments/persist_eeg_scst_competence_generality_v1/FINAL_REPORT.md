# Final report

Terminal: `NO_ADMISSIBLE_COMPETENT_REPRESENTATION_FOUND`

| Model          | Type       |   OpenBMI BA |   OpenBMI 3NN |   WBCIC BA |   WBCIC 3NN | Competent   | All Admissibility Gates   |   SCST Delta BA | CI      | Terminal                            |
|:---------------|:-----------|-------------:|--------------:|-----------:|------------:|:------------|:--------------------------|----------------:|:--------|:------------------------------------|
| EEGNet         | Historical |   nan        |       1.16285 | nan        |     1.30796 | True        | False                     |             nan | NOT RUN | HISTORICAL_CONTROL                  |
| EEGConformer   | Historical |   nan        |       1.14937 | nan        |     1.3408  | True        | False                     |             nan | NOT RUN | HISTORICAL_CONTROL                  |
| CBraMod-frozen | FM         |     0.731    |       1.13113 |   0.750525 |     1.11724 | False       | False                     |             nan | NOT RUN | HISTORICAL_CONTROL                  |
| CBraMod-R1     | FM         |     0.726125 |       1.12671 |   0.746333 |     1.15007 | False       | False                     |             nan | NOT RUN | CBRAMOD_COMPETENCE_NOT_RECOVERED    |
| LaBraM         | FM         |     0.659042 |       1.23739 |   0.753591 |     1.30561 | False       | False                     |             nan | NOT RUN | HISTORICAL_CONTROL                  |
| FBCNet         | Specialist |     0.656708 |     nan       |   0.59562  |   nan       | False       | False                     |             nan | NOT RUN | SPECIALIST_NOT_COMPETENT            |
| ATCNet         | Specialist |     0.767042 |       1.19364 |   0.7859   |     1.27358 | True        | False                     |             nan | NOT RUN | SPECIALIST_COMPETENT_NOT_ADMISSIBLE |
| EEGInceptionMI | Specialist |     0.743167 |     nan       |   0.605991 |   nan       | False       | False                     |             nan | NOT RUN | SPECIALIST_NOT_COMPETENT            |

Strongest supported claim: Under the frozen thresholds and bounded rescue/specialist screen, no representation was both task competent and SCST admissible across OpenBMI and WBCIC.

Most serious limitation: the gate closed before future-session utility evaluation, so this experiment cannot determine whether SCST training improves generalization when admissibility is achieved.
