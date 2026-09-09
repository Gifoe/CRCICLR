# X-only downstream amendment

After inspecting the completed two-MI inner-validation screen, the user
requested that only `LiteBN_X` continue into the remaining tasks and multi-seed
work. This is an outcome-aware downstream reduction and is recorded explicitly;
it must not be presented as the original blind five-architecture protocol.

The exact LiteBN baseline remains in every downstream comparison. `LiteBN-R`,
`LiteBN-RG`, and `LiteBN-XS` are not trained further for ERP/SSVEP or later
seeds. No model definition, preprocessing, subject split, optimizer, epoch
budget, checkpoint rule, or metric is changed. The final held-out/test cohort
remains unopened.
