import numpy as np
import pytest
from hsc_tta.prediction_sets import prediction_sets, evaluate_prediction_sets


def test_prediction_sets_are_nested_and_include_argmax():
    p=np.array([[.6,.3,.1],[.2,.2,.6]]); grid=np.linspace(.5,.99,20)
    sets=prediction_sets(p,grid)
    assert np.all(sets[:,1:,:] >= sets[:,:-1,:])
    assert np.all(sets[np.arange(2),:,p.argmax(1)])
    assert np.all((sets.sum(2)>=1)&(sets.sum(2)<=3))


def test_invalid_probabilities_fail():
    with pytest.raises(ValueError): prediction_sets(np.array([[.6,.6]]),np.array([.5]))


def test_evaluation_is_computed():
    rows=evaluate_prediction_sets(np.array([[.8,.2],[.1,.9]]),np.array([0,1]),np.array([.5,.9]))
    assert rows[0]["future_risk"] == 0
    assert rows[1]["average_set_size"] >= rows[0]["average_set_size"]

