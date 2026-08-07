package com.dragonnest.agent.vendor

import android.content.Context
import android.util.Log
import com.dragonnest.agent.AndroidModelArtifact
import com.dragonnest.agent.AndroidRuntimeBridge
import com.dragonnest.agent.RuntimeExecutionRequest
import com.dragonnest.agent.RuntimeExecutionResult
import com.geniex.sdk.GenieXSdk
import com.geniex.sdk.LlmWrapper
import com.geniex.sdk.bean.ChatMessage
import com.geniex.sdk.bean.ComputeUnitValue
import com.geniex.sdk.bean.GenerationConfig
import com.geniex.sdk.bean.LlmCreateInput
import com.geniex.sdk.bean.LlmStreamResult
import com.geniex.sdk.bean.ModelConfig
import com.geniex.sdk.bean.SamplerConfig
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Physically validated GenieX 0.3.5 / QAIRT 2.45 adapter for the S25 bundles.
 *
 * The SDK dependency and its native closure are packaged only when the explicit
 * S25 hardware-build flag is enabled. A physical runtime error is propagated;
 * this bridge never delegates to DragonNest's mock executor.
 */
class GenieXRuntimeBridge : AndroidRuntimeBridge {
    private companion object {
        const val TAG = "DragonNestGenieX"
    }

    override fun runtimeName(): String = "genie"

    override fun runtimeVersion(): String = "GenieX-0.3.5 / QAIRT-2.45"

    override fun isAvailable(context: Context, artifact: AndroidModelArtifact): Boolean {
        if (artifact.runtime() != "genie" || !GenieXInitialization.await(context)) {
            return false
        }
        return try {
            runBlocking { createWrapper(artifact).destroy() }
            true
        } catch (failure: Throwable) {
            Log.e(TAG, "Execution-ready probe failed for ${artifact.modelId()}", failure)
            false
        }
    }

    override fun execute(
        context: Context,
        artifact: AndroidModelArtifact,
        request: RuntimeExecutionRequest,
    ): RuntimeExecutionResult {
        require(request.inputBoundary() == null) {
            "GenieX full-model bundles do not accept pipeline boundary tensors"
        }
        check(GenieXInitialization.await(context)) { "GenieX initialization failed" }
        return runBlocking {
            val wrapper = createWrapper(artifact)
            try {
                val formatted = wrapper.applyChatTemplate(
                    arrayOf(ChatMessage(role = "user", content = request.requestText())),
                    tools = null,
                    enableThinking = false,
                ).getOrThrow().formattedText
                val text = StringBuilder()
                var completed = false
                wrapper.generateStreamFlow(
                    formatted,
                    GenerationConfig(
                        maxTokens = generationLimit(artifact, request),
                        samplerConfig = SamplerConfig(
                            temperature = 0.01f,
                            topP = 1.0f,
                            topK = 1,
                            seed = 42,
                        ),
                    ),
                ).collect { event ->
                    when (event) {
                        is LlmStreamResult.Token -> text.append(event.text)
                        is LlmStreamResult.Completed -> completed = true
                        is LlmStreamResult.Error -> throw event.throwable
                    }
                }
                check(completed) { "GenieX generation ended without completion" }
                val output = text.toString().trim()
                check(output.isNotBlank()) { "GenieX returned an empty response" }
                RuntimeExecutionResult(output, null, "htp")
            } finally {
                wrapper.destroy()
            }
        }
    }

    private suspend fun createWrapper(artifact: AndroidModelArtifact): LlmWrapper {
        val directory = artifact.artifactPath().toFile()
        check(directory.isDirectory) { "GenieX artifact is not a directory: $directory" }
        val bins = directory.listFiles { file -> file.isFile && file.extension == "bin" }
            ?.sortedBy { it.name }
            .orEmpty()
        check(bins.map { it.name } == listOf("part1_of_2.bin", "part2_of_2.bin")) {
            "GenieX artifact must contain exactly part1_of_2.bin and part2_of_2.bin"
        }
        val tokenizer = File(directory, "tokenizer.json")
        check(tokenizer.isFile) { "GenieX tokenizer is missing: $tokenizer" }
        return LlmWrapper.builder()
            .llmCreateInput(
                LlmCreateInput(
                    model_name = artifact.modelId(),
                    model_path = bins.first().absolutePath,
                    tokenizer_path = tokenizer.absolutePath,
                    config = ModelConfig(nCtx = 0, nGpuLayers = 0, enable_thinking = false),
                    runtime_id = "qairt",
                    compute_unit = ComputeUnitValue.NPU.value,
                ),
            )
            .build()
            .getOrThrow()
    }

    private fun generationLimit(
        artifact: AndroidModelArtifact,
        request: RuntimeExecutionRequest,
    ): Int {
        if (request.maxNewTokens() > 0) {
            return request.maxNewTokens().coerceAtMost(artifact.maxContextTokens())
        }
        val configured = JSONObject(artifact.runtimeOptionsJson()).optInt("max_new_tokens", 96)
        return configured.coerceIn(1, artifact.maxContextTokens())
    }

    private object GenieXInitialization {
        private enum class State { NEW, STARTING, READY, FAILED }

        private val lock = Any()
        private var state = State.NEW
        private var latch = CountDownLatch(1)

        fun await(context: Context): Boolean {
            synchronized(lock) {
                if (state == State.NEW) {
                    state = State.STARTING
                    GenieXSdk.getInstance().init(
                        context.applicationContext,
                        object : GenieXSdk.InitCallback {
                            override fun onSuccess() = finish(State.READY)
                            override fun onFailure(reason: String) {
                                Log.e(TAG, "GenieX initialization failed: $reason")
                                finish(State.FAILED)
                            }
                        },
                    )
                }
                if (state == State.READY) return true
                if (state == State.FAILED) return false
            }
            return latch.await(60, TimeUnit.SECONDS) && synchronized(lock) {
                state == State.READY
            }
        }

        private fun finish(result: State) {
            synchronized(lock) {
                state = result
                latch.countDown()
            }
        }
    }
}
