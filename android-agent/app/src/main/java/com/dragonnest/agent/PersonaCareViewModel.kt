package com.dragonnest.agent

import android.app.ActivityManager
import android.app.Application
import android.content.Intent
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
)

data class ChatUiState(
    val messages: List<ChatMessage> = emptyList(),
    val sending: Boolean = false,
)

class PersonaCareViewModel(application: Application) : AndroidViewModel(application) {
    private val configuration = AgentConfiguration(application)
    private val enrollmentStore = EnrollmentStore(application)
    private val profileStore = UserProfileStore(application)
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
    fun hasEnrollment(): Boolean = enrollmentStore.hasCredential()
    fun currentProfile(): UserProfile? = profileStore.load()

    fun saveConnection(host: String, portText: String, code: String, useTls: Boolean) {
        val normalizedHost = host.trim()
        val port = portText.toIntOrNull()
            ?: throw IllegalArgumentException("Enter a valid port")
        if (normalizedHost.isBlank() || normalizedHost.any { it.isWhitespace() }) {
            throw IllegalArgumentException("Enter the DragonNest server address")
        }
        if (port !in 1..65535) {
            throw IllegalArgumentException("Port must be between 1 and 65535")
        }
        if (code.isBlank() && !enrollmentStore.hasCredential()) {
            throw IllegalArgumentException("Enter the enrollment code")
        }
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

    fun saveProfile(name: String, about: String, personaId: String) {
        val profile = UserProfile(name, about, personaId)
        profileStore.save(profile)
        val start = Intent(getApplication(), AgentForegroundService::class.java).apply {
            action = AgentForegroundService.ACTION_RELOAD
        }
        getApplication<Application>().startForegroundService(start)
    }

    fun submit(
        prompt: String,
        personaId: String,
        useProfile: Boolean,
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
            val response = runCatching {
                withContext(Dispatchers.IO) {
                    BrainTaskClient(configuration).submit(
                        text,
                        personaId,
                        useProfile,
                        computePreference.wireValue,
                    )
                }
            }
            val message = response.fold(
                onSuccess = ::messageFromResponse,
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

    private fun messageFromResponse(response: SubmitTaskResponse): ChatMessage {
        if (response.success) {
            return ChatMessage(
                System.nanoTime(),
                false,
                response.outputText,
                response.deviceDisplayName.ifBlank { response.deviceId },
            )
        }
        val message = when (response.errorCode) {
            "STEERING_UNAVAILABLE" -> "That persona is not available on a connected model."
            "NO_ELIGIBLE_FALLBACK" -> "No compatible DragonNest device is ready."
            "LOCAL_UNAVAILABLE" -> "Local compute is not available with current device resources."
            "ELASTIC_UNAVAILABLE" -> "Elastic compute is not available on the connected devices yet."
            "ORIGIN_DEVICE_REQUIRED" -> "This request cannot run privately yet."
            else -> response.errorMessage.ifBlank { "DragonNest could not complete the request." }
        }
        return ChatMessage(System.nanoTime(), false, message, failed = true)
    }
}
