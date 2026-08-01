from hsc_tta.protocols import choose_common_central_channel


def test_channel_choice_uses_availability_and_c4_tie_break_only():
    availability = {
        "hmc": {"C3": ["h1", "h2"], "C4": ["h1", "h2"]},
        "cap": {"C3": ["c1"], "C4": ["c1"]},
    }
    protocol = choose_common_central_channel(availability)
    assert protocol["selected_channel"] == "C4"
    assert protocol["selection_basis"] == "channel_availability_only"
    assert len(protocol["protocol_hash"]) == 64
