"""The numbers the article quotes, checked against the code."""

from pytest import approx

from ai_04_forward_pass.answer import Gap
from ai_04_forward_pass.mango import MANGO, NETWORK, RIPE
from ai_04_forward_pass.matrix import MatrixNetwork, same_values
from ai_04_forward_pass.work import CPU_CORE, DIGIT_READER, GPU, multiplications_per_layer

RUN = NETWORK.forward(MANGO)
H1, H2, H3, Y = RUN.workings


def test_the_network_has_13_parameters():
    assert (NETWORK.weight_count, NETWORK.bias_count) == (9, 4)
    assert NETWORK.parameter_count == 13


def test_h1_is_0_7():
    assert H1.products == approx((0.8, -0.2))
    assert H1.z == approx(0.7)
    assert H1.value == approx(0.7)


def test_h2_is_0_55():
    assert H2.products == approx((0.3, 0.45))
    assert H2.z == approx(0.55)
    assert H2.value == approx(0.55)


def test_h3_sums_to_minus_0_05_and_is_switched_off():
    assert H3.z == approx(-0.05)
    assert H3.value == 0
    assert H3.switched_off


def test_the_output_is_0_59():
    assert Y.products == approx((0.84, -0.385, 0))
    assert Y.z == approx(0.355)
    assert round(RUN.answer, 2) == 0.59


def test_the_matrix_pass_gives_the_same_answer():
    by_matrix = MatrixNetwork.from_network(NETWORK).forward(MANGO)
    hidden, output = by_matrix
    assert hidden == approx([0.7, 0.55, 0])
    assert output == approx([RUN.answer])
    assert same_values(RUN, by_matrix)


def test_the_pass_does_9_multiplications():
    per_layer = multiplications_per_layer(NETWORK.layer_sizes)
    assert per_layer == [6, 3]
    assert per_layer == [sum(len(w.products) for w in layer) for layer in RUN.layers]


def test_one_cpu_core_takes_9_steps_and_a_gpu_2():
    per_layer = multiplications_per_layer(NETWORK.layer_sizes)
    assert len(CPU_CORE.schedule(per_layer)) == 9
    assert len(GPU.schedule(per_layer)) == 2


def test_the_digit_reader_does_213472_multiplications():
    assert sum(multiplications_per_layer(DIGIT_READER)) == 213_472


def test_the_answer_is_0_41_short():
    gap = Gap(guess=RUN.answer, right_answer=RIPE)
    assert round(gap.short_by, 2) == 0.41
