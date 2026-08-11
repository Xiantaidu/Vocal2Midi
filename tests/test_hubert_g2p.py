from inference.HubertFA.tools.g2p import JapanesePhonemeMoraG2P


def test_japanese_global_cl_phoneme_is_not_language_prefixed():
    phonemes, words, _ = JapanesePhonemeMoraG2P("ja")("cl")

    assert phonemes == ["SP", "cl", "SP"]
    assert words == ["cl"]
