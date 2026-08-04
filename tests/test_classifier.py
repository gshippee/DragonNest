from dragon_nest.classifier import RuleBasedTaskClassifier


def test_classifier_marks_compound_parallel_and_steering():
    profile = RuleBasedTaskClassifier().classify(
        "Summarize section 1 and also section 2 in a concise tone.",
        preferred_mode="parallel",
    )
    assert profile.task_class == "summarization"
    assert profile.data_parallelizable
    assert profile.steering_requested
    assert profile.is_compound


def test_private_mode_sets_device_only_privacy():
    profile = RuleBasedTaskClassifier().classify("Rewrite this note.", preferred_mode="private")
    assert profile.privacy_tier == "device_only"

