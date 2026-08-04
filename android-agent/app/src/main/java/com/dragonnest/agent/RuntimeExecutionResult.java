package com.dragonnest.agent;

import com.dragonnest.proto.BoundaryTensor;

/** Output produced by a packaged QNN or Genie runtime bridge. */
public record RuntimeExecutionResult(
        String outputText,
        BoundaryTensor boundary,
        String accelerator) {
    public RuntimeExecutionResult {
        outputText = outputText == null ? "" : outputText;
        accelerator = accelerator == null ? "" : accelerator;
    }
}
