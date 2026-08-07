package com.dragonnest.agent;

import com.dragonnest.proto.BoundaryTensor;

/** Output produced by a packaged QNN or Genie runtime bridge. */
public record RuntimeExecutionResult(
        String outputText,
        BoundaryTensor boundary,
        String accelerator,
        Integer nextTokenId,
        boolean eos,
        String tokenText) {
    public RuntimeExecutionResult {
        outputText = outputText == null ? "" : outputText;
        accelerator = accelerator == null ? "" : accelerator;
        tokenText = tokenText == null ? "" : tokenText;
    }

    public RuntimeExecutionResult(
            String outputText, BoundaryTensor boundary, String accelerator) {
        this(outputText, boundary, accelerator, null, false, "");
    }
}
