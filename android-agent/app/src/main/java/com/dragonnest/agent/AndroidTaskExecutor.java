package com.dragonnest.agent;

import com.dragonnest.proto.ExecutePipelineStage;
import com.dragonnest.proto.ExecuteShard;
import com.dragonnest.proto.ExecuteTask;

public interface AndroidTaskExecutor {
    TaskExecutionResult execute(ExecuteTask command) throws Exception;

    TaskExecutionResult executeShard(ExecuteShard command) throws Exception;

    TaskExecutionResult executePipelineStage(ExecutePipelineStage command) throws Exception;
}
