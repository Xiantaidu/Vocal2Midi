"""TiFA (Token-Imputing Forced Aligner) integration.

Vendored G2P pipeline and host-side ONNX inference for the openvpi/TIFA
aligner (MIT license). The g2p subtree and lib/{vocabulary,levenshtein} come
from https://github.com/openvpi/TIFA; host-side ONNX orchestration, Viterbi
and dictionary loading are adapted from the same repository for Vocal2Midi.
"""
