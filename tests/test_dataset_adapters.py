from hsc_tta.data import HMCAdapter,CAPAdapter,EEGMMIDBAdapter


def test_stable_dataset_prefixed_subject_ids(tmp_path):
    assert HMCAdapter(tmp_path).subject_id(tmp_path/"SN001.edf")=="hmc:001"
    assert CAPAdapter(tmp_path).subject_id(tmp_path/"nfle1.edf").startswith("cap:")
    assert EEGMMIDBAdapter(tmp_path).subject_id(tmp_path/"S001R04.edf")=="eegmmidb:001"

