import numpy as np

from shadow_zone.data.sample import load_sample
from shadow_zone.model import FramingModel


def test_sample_strike_rate_falls_through_the_edge():
    d = load_sample()
    inside, outside = d.loc[d["edge_dist"].between(-0.3, -0.05)], d.loc[d["edge_dist"].between(0.05, 0.3)]
    assert inside["strike"].mean() > 0.8 > 0.3 > outside["strike"].mean() > 0.0
    assert d.loc[d["edge_dist"] > 0.8, "strike"].mean() < 0.02
    assert d.loc[d["edge_dist"] < -0.5, "strike"].mean() > 0.97


def test_sample_fit_and_predict_run_offline():
    d = load_sample()
    m = FramingModel(scales={"catcher": 0.15, "umpire": 0.15, "pitcher": 0.3, "batter": 0.2}).fit(d)
    p = m.predict(d)
    assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
    assert np.corrcoef(p, d["strike"])[0, 1] > 0.85
    assert abs(m.effects("catcher")["effect_logit"].mean()) < 0.1
