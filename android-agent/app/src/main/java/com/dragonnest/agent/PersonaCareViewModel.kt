package com.dragonnest.agent

import android.app.ActivityManager
import android.app.Application
import android.content.Intent
import android.media.AudioAttributes
import android.media.MediaPlayer
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.dragonnest.proto.SubmitTaskResponse
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class ChatMessage(
    val id: Long,
    val fromUser: Boolean,
    val text: String,
    val deviceName: String = "",
    val failed: Boolean = false,
    val routeSummary: String = "",
    val profileSummary: String = "",
)

data class ChatUiState(
    val messages: List<ChatMessage> = emptyList(),
    val sending: Boolean = false,
    // Speech is per-message: the reply being synthesized on the Brain's NPU,
    // and the one currently playing back through the speaker.
    val synthesizingMessageId: Long? = null,
    val speakingMessageId: Long? = null,
    val speechError: String? = null,
)

class PersonaCareViewModel(application: Application) : AndroidViewModel(application) {
    private val configuration = AgentConfiguration(application)
    private val enrollmentStore = EnrollmentStore(application)
    private val profileStore = UserProfileStore(application)
    private val speechClient = SpeechClient(application)
    private var mediaPlayer: MediaPlayer? = null
    private val mutableChat = MutableStateFlow(ChatUiState())
    val chat: StateFlow<ChatUiState> = mutableChat.asStateFlow()

    val agentStatus = AgentStatusRepository.status

    init {
        if (enrollmentStore.hasCredential() && profileStore.load() != null) {
            val start = Intent(application, AgentForegroundService::class.java).apply {
                action = AgentForegroundService.ACTION_RELOAD
            }
            application.startForegroundService(start)
        }
    }

    val initialRoute: String
        get() = when {
            !enrollmentStore.hasCredential() -> "connect"
            profileStore.load() == null -> "profile"
            else -> "chat"
        }

    fun currentHost(): String = configuration.brainHost()
    fun currentPort(): String = configuration.brainPort().toString()
    fun currentTls(): Boolean = configuration.useTls()
    fun currentDashboardPort(): String = configuration.dashboardPort().toString()
    fun hasEnrollment(): Boolean = enrollmentStore.hasCredential()
    fun currentProfile(): UserProfile? = profileStore.load()

    fun saveConnection(
        host: String,
        portText: String,
        code: String,
        useTls: Boolean,
        dashboardPortText: String = currentDashboardPort(),
    ) {
        val normalizedHost = host.trim()
        val port = portText.toIntOrNull()
            ?: throw IllegalArgumentException("Enter a valid port")
        val dashboardPort = dashboardPortText.toIntOrNull()
            ?: throw IllegalArgumentException("Enter a valid dashboard port")
        if (normalizedHost.isBlank() || normalizedHost.any { it.isWhitespace() }) {
            throw IllegalArgumentException("Enter the DragonNest server address")
        }
        if (port !in 1..65535) {
            throw IllegalArgumentException("Port must be between 1 and 65535")
        }
        if (dashboardPort !in 1..65535) {
            throw IllegalArgumentException("Dashboard port must be between 1 and 65535")
        }
        if (code.isBlank() && !enrollmentStore.hasCredential()) {
            throw IllegalArgumentException("Enter the enrollment code")
        }
        configuration.saveDashboardPort(dashboardPort)
        configuration.saveEnrollmentEndpoint(normalizedHost, port, useTls)
        if (code.isNotBlank()) enrollmentStore.save(code.trim())
        reloadAgentIfReady()
    }

    fun applyEnrollment(payload: EnrollmentPayload) {
        configuration.saveEnrollmentEndpoint(
            payload.brainHost(),
            payload.brainPort(),
            payload.useTls(),
        )
        enrollmentStore.save(payload.credential())
        reloadAgentIfReady()
    }

    fun saveProfile(
        name: String,
        about: String,
        personaId: String,
        steeringAlpha: Float = 0f,
    ) {
        val profile = UserProfile(name, about, personaId, steeringAlpha)
        profileStore.save(profile)
        val start = Intent(getApplication(), AgentForegroundService::class.java).apply {
            action = AgentForegroundService.ACTION_RELOAD
        }
        getApplication<Application>().startForegroundService(start)
    }

    fun submit(
        prompt: String,
        computePreference: ComputePreference,
    ) {
        val text = prompt.trim()
        if (text.isEmpty() || mutableChat.value.sending) return
        val userMessage = ChatMessage(System.nanoTime(), true, text)
        mutableChat.value = mutableChat.value.copy(
            messages = mutableChat.value.messages + userMessage,
            sending = true,
        )
        viewModelScope.launch {
            val storedProfile = profileStore.load()
            val personaId = storedProfile?.personaId() ?: UserProfile.PERSONA_BALANCED
            val steeringAlpha = storedProfile?.steeringAlpha() ?: 0f
            val response = runCatching {
                withContext(Dispatchers.IO) {
                    BrainTaskClient(configuration).submit(
                        text,
                        personaId,
                        true,
                        computePreference.wireValue,
                        steeringAlpha,
                    )
                }
            }
            val message = response.fold(
                onSuccess = { messageFromResponse(it, computePreference, personaId) },
                onFailure = {
                    ChatMessage(
                        System.nanoTime(),
                        false,
                        "PersonaCare could not reach DragonNest. Try again.",
                        failed = true,
                    )
                },
            )
            mutableChat.value = mutableChat.value.copy(
                messages = mutableChat.value.messages + message,
                sending = false,
            )
        }
    }

    /**
     * Start a fresh conversation. Any reply being read aloud is stopped first:
     * the audio belongs to a message that is about to disappear, so letting it
     * keep playing would leave the speaker running with nothing on screen to
     * stop it.
     */
    fun newChat() {
        stopSpeech()
        mutableChat.value = ChatUiState()
    }

    /**
     * Read a reply aloud. The text is sent to the Brain, which runs MeloTTS on
     * its own NPU and returns a .wav -- this phone's HTP cannot load that
     * model, whose context binaries are compiled for Snapdragon X Elite.
     * Tapping again while it plays stops playback.
     */
    fun speak(message: ChatMessage) {
        if (message.fromUser || message.failed || message.text.isBlank()) return
        if (mutableChat.value.speakingMessageId == message.id) {
            stopSpeech()
            return
        }
        if (mutableChat.value.synthesizingMessageId != null) return
        stopSpeech()
        mutableChat.value = mutableChat.value.copy(
            synthesizingMessageId = message.id,
            speechError = null,
        )
        viewModelScope.launch {
            val audio = runCatching {
                withContext(Dispatchers.IO) { speechClient.synthesize(message.text) }
            }
            audio.fold(
                onSuccess = { play(message.id, it) },
                onFailure = {
                    mutableChat.value = mutableChat.value.copy(
                        synthesizingMessageId = null,
                        speechError = it.message ?: "DragonNest could not read this aloud.",
                    )
                },
            )
        }
    }

    private fun play(messageId: Long, audio: java.io.File) {
        val player = MediaPlayer()
        runCatching {
            player.setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build(),
            )
            player.setDataSource(audio.path)
            player.setOnCompletionListener { stopSpeech() }
            player.setOnErrorListener { _, _, _ ->
                mutableChat.value = mutableChat.value.copy(
                    speechError = "That audio could not be played.",
                )
                stopSpeech()
                true
            }
            player.prepare()
            player.start()
        }.onFailure {
            player.release()
            mutableChat.value = mutableChat.value.copy(
                synthesizingMessageId = null,
                speechError = "That audio could not be played.",
            )
            return
        }
        mediaPlayer = player
        mutableChat.value = mutableChat.value.copy(
            synthesizingMessageId = null,
            speakingMessageId = messageId,
        )
    }

    fun stopSpeech() {
        mediaPlayer?.let { player ->
            runCatching { if (player.isPlaying) player.stop() }
            player.release()
        }
        mediaPlayer = null
        mutableChat.value = mutableChat.value.copy(speakingMessageId = null)
    }

    /** Clear a speech error once it has been shown, so it is not repeated. */
    fun consumeSpeechError() {
        if (mutableChat.value.speechError != null) {
            mutableChat.value = mutableChat.value.copy(speechError = null)
        }
    }

    override fun onCleared() {
        stopSpeech()
        super.onCleared()
    }

    fun totalMemoryMb(): Long {
        val manager = getApplication<Application>().getSystemService(ActivityManager::class.java)
        val info = ActivityManager.MemoryInfo()
        manager.getMemoryInfo(info)
        return info.totalMem / (1024L * 1024L)
    }

    fun currentSimulatedMemoryMb(): Long? = configuration.simulatedMemoryMb()

    fun setSimulatedMemoryMb(memoryMb: Long?) {
        configuration.saveSimulatedMemoryMb(memoryMb)
        val update = Intent(getApplication(), AgentForegroundService::class.java).apply {
            action = AgentForegroundService.ACTION_UPDATE_SIMULATION
        }
        getApplication<Application>().startForegroundService(update)
    }

    fun clearLocalRegistration() {
        getApplication<Application>().stopService(
            Intent(getApplication(), AgentForegroundService::class.java),
        )
        enrollmentStore.clear()
        configuration.clearEnrollmentEndpoint()
        AgentStatusRepository.update(AgentConnectionState.IDLE, "")
    }

    private fun reloadAgentIfReady() {
        if (profileStore.load() == null) return
        val start = Intent(getApplication(), AgentForegroundService::class.java).apply {
            action = AgentForegroundService.ACTION_RELOAD
        }
        getApplication<Application>().startForegroundService(start)
    }

    private fun messageFromResponse(
        response: SubmitTaskResponse,
        computePreference: ComputePreference,
        requestedPersonaId: String,
    ): ChatMessage {
        if (response.success) {
            val realization = when (response.steering.mode) {
                "baked_profile" -> "baked"
                "prompt_profile" -> "prompt-conditioned"
                "runtime_vector" -> if (response.steering.enabled) "runtime vector" else "base"
                else -> "base"
            }
            val profileId = response.steering.behaviorProfileId
                .ifBlank { requestedPersonaId }
            return ChatMessage(
                System.nanoTime(),
                false,
                response.outputText,
                response.deviceDisplayName.ifBlank { response.deviceId },
                routeSummary = "${computePreference.displayName} · ${displayModel(response.modelId)}",
                profileSummary = "Profile: ${profileId.replaceFirstChar { it.uppercase() }} · $realization",
            )
        }
        val message = when (response.errorCode) {
            "PROFILE_UNAVAILABLE" -> "That response style is not installed on an eligible device yet."
            "STEERING_UNAVAILABLE" -> "That persona is not available on a connected model."
            "NO_ELIGIBLE_FALLBACK" -> "No compatible DragonNest device is ready."
            "LOCAL_UNAVAILABLE" -> "Local compute is not available with current device resources."
            "ELASTIC_UNAVAILABLE" -> "Elastic compute is not available on the connected devices yet."
            "ORIGIN_DEVICE_REQUIRED" -> "This request cannot run privately yet."
            else -> response.errorMessage.ifBlank { "DragonNest could not complete the request." }
        }
        return ChatMessage(System.nanoTime(), false, message, failed = true)
    }

    private fun displayModel(modelId: String): String = when {
        modelId.startsWith("qwen3-0.6b") -> "Qwen3-0.6B"
        modelId.startsWith("qwen3-1.7b") -> "Qwen3-1.7B"
        modelId.startsWith("qwen3-4b") -> "Qwen3-4B"
        modelId.isBlank() -> "model pending"
        else -> modelId
    }
}
