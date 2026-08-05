import json

import pytest

from hsc_tta.contextual_risk.access import ContextualAccessController


def _payload():
    return {"subject_id":"hmc:001","dataset":"hmc","seed":0,"role":"formal_calibration","selected_branch":"A","alpha":.1,"delta":.1,"context_hash":"c","source_model_hash":"m","method_config_hash":"x","feature_hash":"f","certified_index":7}


def test_future_is_closed_until_valid_decision_is_frozen(tmp_path):
    controller=ContextualAccessController("hmc:001","hmc",0,"formal_calibration")
    with pytest.raises(RuntimeError): controller.open_future([1],tmp_path/"d.json")
    path=tmp_path/"d.json";controller.freeze_decision(_payload(),path)
    assert controller.open_future([1,2],path)==[1,2]


def test_decision_tampering_is_detected(tmp_path):
    controller=ContextualAccessController("hmc:001","hmc",0,"formal_calibration")
    path=tmp_path/"d.json";controller.freeze_decision(_payload(),path)
    payload=json.loads(path.read_text());payload["certified_index"]=0;path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError): controller.open_future([1],path)
