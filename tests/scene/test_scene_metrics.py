"""Every metric reports its denominator, so a vacuous score cannot pass as a good one."""
from eval.scene_metrics import (avg_llm_calls_per_event, scene_false_consent_rate,
                                scene_false_speech_rate, scene_recall)

SPOKE_WHEN_SILENT = [{"expect": "", "actual": "话"}, {"expect": "", "actual": ""}]
SHOULD_HAVE_SPOKEN = [{"expect": "问", "actual": "问"}, {"expect": "问", "actual": ""}]


def test_false_speech_counts_only_rows_gold_says_are_silent():
    assert scene_false_speech_rate(SPOKE_WHEN_SILENT) == 0.5


def test_false_speech_ignores_rows_that_were_supposed_to_speak():
    """A row gold says should speak cannot be a false positive, however it turned out."""
    assert scene_false_speech_rate(SHOULD_HAVE_SPOKEN) == 0.0


def test_recall_counts_only_rows_gold_says_should_speak():
    assert scene_recall(SHOULD_HAVE_SPOKEN) == 0.5


def test_recall_requires_the_right_sentence_not_merely_some_sentence():
    assert scene_recall([{"expect": "问", "actual": "别的话"}]) == 0.0


def test_false_consent_counts_rows_that_must_not_consent():
    rows = [{"expect_consent": False, "consented": True},
            {"expect_consent": False, "consented": False}]
    assert scene_false_consent_rate(rows) == 0.5


def test_false_consent_ignores_rows_where_consent_was_correct():
    assert scene_false_consent_rate([{"expect_consent": True, "consented": True}]) == 0.0


def test_an_empty_denominator_is_zero_not_a_crash():
    """A vacuous 1.0 reads as success. Every metric reports 0.0 on no rows, and the runner
    prints the denominator beside it so the emptiness is visible."""
    for fn in (scene_false_speech_rate, scene_recall, scene_false_consent_rate,
               avg_llm_calls_per_event):
        assert fn([]) == 0.0
