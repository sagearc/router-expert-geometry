"""Immutable OLMoE checkpoints used by the retained public-model experiments."""

from typing import Final

# Immutable revisions shared by response capture and steering evaluation.
OLMOE_CHECKPOINT_REVISIONS: Final[dict[int, str]] = {
    5_000: '37c50a5f4d30fba036185f57168ee8df0003f2e0',
    10_000: 'e5642d2d7b37c138775ddf59f685f91788e173b7',
    15_000: '7e6a08024846e9da8c26a68a5d0d5d2182b6b14d',
    20_000: '2c10f9b2376b3952faaeec9bceb499c6386f1e21',
    25_000: '9a1652621aead9261ed960984cedc57544566828',
    50_000: 'a4cc2707a1c90cf72a30ea3c50f679680ee290b5',
    100_000: '9774e8a4aaeff0b25dd2fa7548ee3d38df7c8600',
    250_000: '64060e961bc36ccbff5de271e6efa5e6f10acd2b',
    500_000: '847a3441998b90b3b55323eaa47e2ca6a039bb39',
    1_000_000: '33d21fb2f1382acb824dcdf25e893f44f3ad57ed',
    1_220_000: '811a27549703ad99a2f7303da68cc720c242b456',
}
