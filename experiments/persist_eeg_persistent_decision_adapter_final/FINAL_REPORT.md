# Final report

* Selected recipe: `pda_r1_lx0.5_lp0.01` (rank 1, lambda_X 0.5, lambda_P 0.01).
* OpenBMI source full-vs-population: ΔBA=-0.0436, 95% CI [-0.0618, -0.0267], n=40.
* WBCIC source full-vs-population: ΔBA=-0.0118, 95% CI [-0.0204, -0.0036], n=41.
* Ordinary adapter: OpenBMI ΔBA=-0.0135, 95% CI [-0.0309, +0.0032], n=40; WBCIC ΔBA=-0.0013, 95% CI [-0.0048, +0.0020], n=41.
* Single-session: OpenBMI ΔBA=+0.0763, 95% CI [+0.0581, +0.0943], n=40; WBCIC ΔBA=+0.0079, 95% CI [+0.0012, +0.0162], n=41.
* Correct-vs-wrong: OpenBMI ΔBA=-0.0014, 95% CI [-0.0153, +0.0114], n=40; WBCIC ΔBA=+0.0018, 95% CI [-0.0051, +0.0096], n=41.
* Correct-vs-shuffled: OpenBMI ΔBA=+0.0097, 95% CI [-0.0104, +0.0288], n=40; WBCIC ΔBA=+0.0035, 95% CI [-0.0056, +0.0127], n=41.
* ATCNet WBCIC S2: NOT RUN (source gate failed).
* EEGNeX: NOT RUN (ATCNet source gate failed).
* Outer: SEALED / NOT INSPECTED.

Strongest supported claim: source-only PDA did not satisfy the preregistered
transfer and mechanism gate on these frozen representations. Exact terminal:
`PERSIST_PDA_SOURCE_NOT_SUPPORTED`.
