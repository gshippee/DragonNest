from __future__ import annotations

import re

from .models import TaskProfile


class RuleBasedTaskClassifier:
    """Small deterministic classifier for routing-relevant task shape."""

    def classify(self, request_text: str, preferred_mode: str = "auto") -> TaskProfile:
        text = request_text.lower()
        task_class = "chat_qa"
        confidence = 0.60

        rules = [
            (("summarize", "summary", "key points"), "summarization", 0.86),
            (("extract", "action items", "find all", "parse"), "document_extraction", 0.84),
            (("compare", "recommend", "trade-offs", "analyze"), "reasoning_analysis", 0.88),
            (("rewrite", "shorten", "translate", "tone"), "translation_rewrite", 0.83),
            (("code", "function", "stack trace", "bug"), "code_assistance", 0.85),
        ]
        for triggers, klass, score in rules:
            if any(trigger in text for trigger in triggers):
                task_class = klass
                confidence = score
                break

        estimated_input_tokens = max(1, len(request_text.split()) * 4 // 3)
        estimated_output_tokens = 128
        if task_class in {"reasoning_analysis", "code_assistance"}:
            estimated_output_tokens = 384
        if task_class == "summarization":
            estimated_output_tokens = 220

        compound_markers = (" then ", " and also ", " for each ", "\n1.", "\n2.", ";")
        numbered_items = len(re.findall(r"(^|\n)\s*\d+[.)]", request_text))
        is_compound = any(marker in text for marker in compound_markers) or numbered_items >= 2

        reasoning_words = ("compare", "recommend", "why", "trade-off", "analyze", "evaluate")
        high_complexity = (
            estimated_input_tokens > 350
            or is_compound
            or any(word in text for word in reasoning_words)
            or preferred_mode == "quality"
        )
        complexity = "high" if high_complexity else "medium" if estimated_input_tokens > 80 else "low"

        privacy_tier = "device_only" if preferred_mode == "private" else "trusted_fabric"
        latency_tier = "realtime" if preferred_mode == "fast" else "interactive"
        steering_requested = any(word in text for word in ("concise", "verbose", "persona", "style", "tone"))
        data_parallelizable = is_compound or " for each " in text or "sections" in text
        layer_parallel_candidate = preferred_mode == "quality" and complexity == "high"

        return TaskProfile(
            task_class=task_class,
            complexity=complexity,
            privacy_tier=privacy_tier,
            latency_tier=latency_tier,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            confidence=confidence,
            is_compound=is_compound,
            data_parallelizable=data_parallelizable,
            layer_parallel_candidate=layer_parallel_candidate,
            steering_requested=steering_requested,
        )

