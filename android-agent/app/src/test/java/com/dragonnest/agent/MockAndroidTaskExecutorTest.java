package com.dragonnest.agent;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.dragonnest.proto.ExecutePipelineStage;
import com.dragonnest.proto.ExecuteShard;
import com.dragonnest.proto.ExecuteTask;

import org.junit.Test;

public final class MockAndroidTaskExecutorTest {
    private final MockAndroidTaskExecutor executor = new MockAndroidTaskExecutor();

    @Test
    public void executesSingleAndShardCommands() {
        TaskExecutionResult single = executor.execute(ExecuteTask.newBuilder()
                .setTaskId("task-1")
                .setAttemptId("attempt-1")
                .setModelId("android-mock-v1")
                .setRequestText("hello")
                .build());
        TaskExecutionResult shard = executor.executeShard(ExecuteShard.newBuilder()
                .setTaskId("task-2")
                .setAttemptId("attempt-2")
                .setShardId("shard-1")
                .setModelId("android-mock-v1")
                .setRequestText("summarize")
                .build());

        assertTrue(single.success());
        assertTrue(single.outputText().contains("Android mock result"));
        assertTrue(shard.success());
        assertTrue(shard.outputText().contains("shard-1"));
    }

    @Test
    public void emitsChecksummedPipelineBoundaryAndFinalOutput() throws Exception {
        TaskExecutionResult first = executor.executePipelineStage(
                ExecutePipelineStage.newBuilder()
                        .setTaskId("task-pipeline")
                        .setAttemptId("attempt-a")
                        .setStageId("stage-1")
                        .setModelId("android-mock-v1")
                        .build());

        assertTrue(first.success());
        assertNotNull(first.boundary());
        assertEquals("hidden", first.boundary().getTensorName());
        assertTrue(first.boundary().getChecksum().startsWith("sha256:"));
        assertEquals(
                first.boundary().getData().size(),
                first.boundary().getShape(0));

        TaskExecutionResult last = executor.executePipelineStage(
                ExecutePipelineStage.newBuilder()
                        .setTaskId("task-pipeline")
                        .setAttemptId("attempt-b")
                        .setStageId("stage-2")
                        .setModelId("android-mock-v1")
                        .setInputBoundary(first.boundary())
                        .setFinalStage(true)
                        .build());

        assertTrue(last.success());
        assertTrue(last.outputText().contains(first.boundary().getChecksum()));
    }
}
