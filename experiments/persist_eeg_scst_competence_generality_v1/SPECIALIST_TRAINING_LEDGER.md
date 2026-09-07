# Specialist training ledger

Selection used source model-fit/validation only; future sessions were not accessed.

## FBCNet — OpenBMI

- Implementation: frozen clean-room canonical family in `code/specialist_models.py`.
- Parameter count: 5186.
- Grid: recipe_a (lr=1e-3, wd=1e-3, 30 epochs) versus recipe_b (lr=3e-4, wd=1e-2, 36 epochs).
- Selected: recipe_a using mean source-validation BA=0.650500; NLL=0.642640.

## FBCNet — WBCIC

- Implementation: frozen clean-room canonical family in `code/specialist_models.py`.
- Parameter count: 4898.
- Grid: recipe_a (lr=1e-3, wd=1e-3, 30 epochs) versus recipe_b (lr=3e-4, wd=1e-2, 36 epochs).
- Selected: recipe_a using mean source-validation BA=0.599428; NLL=0.696151.

## ATCNet — OpenBMI

- Implementation: frozen clean-room canonical family in `code/specialist_models.py`.
- Parameter count: 21698.
- Grid: recipe_a (lr=1e-3, wd=1e-3, 30 epochs) versus recipe_b (lr=3e-4, wd=1e-2, 36 epochs).
- Selected: recipe_a using mean source-validation BA=0.746500; NLL=0.549660.

## ATCNet — WBCIC

- Implementation: frozen clean-room canonical family in `code/specialist_models.py`.
- Parameter count: 21570.
- Grid: recipe_a (lr=1e-3, wd=1e-3, 30 epochs) versus recipe_b (lr=3e-4, wd=1e-2, 36 epochs).
- Selected: recipe_a using mean source-validation BA=0.792606; NLL=0.430301.

## EEGInceptionMI — OpenBMI

- Implementation: frozen clean-room canonical family in `code/specialist_models.py`.
- Parameter count: 47170.
- Grid: recipe_a (lr=1e-3, wd=1e-3, 30 epochs) versus recipe_b (lr=3e-4, wd=1e-2, 36 epochs).
- Selected: recipe_a using mean source-validation BA=0.744125; NLL=0.576401.

## EEGInceptionMI — WBCIC

- Implementation: frozen clean-room canonical family in `code/specialist_models.py`.
- Parameter count: 46978.
- Grid: recipe_a (lr=1e-3, wd=1e-3, 30 epochs) versus recipe_b (lr=3e-4, wd=1e-2, 36 epochs).
- Selected: recipe_a using mean source-validation BA=0.611704; NLL=0.671990.

