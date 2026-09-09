# FINAL_SEED0_DECISION

## Backbone competence

See `BACKBONE_COMPETENCE_SUMMARY.csv`; competence was frozen using inner-validation BA before outer reveal.

## Outcome-blind actionability

| Dataset | fresh eligible pairs | rho EC | rho Gdev | EC-selected gain | random gain | oracle gain |
|---|---:|---:|---:|---:|---:|---:|
| OpenBMI | 0 | NA | NA | NA | NA | NA |
| WBCIC | 0 | NA | NA | NA | NA | NA |

## Multi-backbone

See `MULTIBACKBONE_MATRIX.csv` for all 15 fixed pairs, including historical pairs.

## Answers

1. EC 是否能提前 rank actual fusion utility？见 rho EC；本轮只把 pair family 当作 ranking unit。
2. Development fusion gain 是否更能预测未来 utility？见 rho Gdev。
3. 是否存在真实 positive oracle opportunity？见 oracle gain 与 oracle_minus_random。
4. EC-selected pair 是否优于 random？见表。
5. EEGNet+LiteBN 是否仍是特殊 positive case？该 pair 保留为 HISTORICAL，未进入 fresh statistic。
6. 现代 FM pair 是否表现出稳定 generality？由完整矩阵和 competence 结果判断，不能由单一正例外推。
7. seed0 是否值得扩展 seed1/2？只根据本轮冻结规则给出，不自动扩展。

Final terminal: **INCONCLUSIVE**

Claim boundary: any oracle value is recoverable conditional headroom, not a deployable gate; fixed fusion does not solve subject reliability by itself.

Next action (exactly one): stop or treat this seed0 screen as inconclusive; do not claim actionability
