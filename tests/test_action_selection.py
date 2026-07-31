import pandas as pd
from hsc_tta.selection import select_safe_action


def test_lexicographic_selection_and_uncertified():
    f=pd.DataFrame([
      {"action":"t3a","lambda":.8,"certified_upper_bound":.15,"average_set_size":2.,"singleton_rate":.5,"n_classes":5},
      {"action":"no_tta","lambda":.8,"certified_upper_bound":.15,"average_set_size":2.,"singleton_rate":.5,"n_classes":5},
      {"action":"no_tta","lambda":.9,"certified_upper_bound":.15,"average_set_size":2.,"singleton_rate":.5,"n_classes":5},])
    out=select_safe_action(f,.2)
    assert (out["selected_action"],out["selected_lambda"]) == ("no_tta",.9)
    assert select_safe_action(f,.1)["status"] == "uncertified"

