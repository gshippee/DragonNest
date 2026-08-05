package com.dragonnest.agent

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

enum class AgentConnectionState {
    IDLE,
    CONNECTING,
    CONNECTED,
    RETRYING,
    STOPPED,
}

data class AgentStatus(
    val state: AgentConnectionState = AgentConnectionState.IDLE,
    val detail: String = "",
)

object AgentStatusRepository {
    private val mutableStatus = MutableStateFlow(AgentStatus())
    val status: StateFlow<AgentStatus> = mutableStatus.asStateFlow()

    @JvmStatic
    fun update(state: AgentConnectionState, detail: String = "") {
        mutableStatus.value = AgentStatus(state, detail)
    }
}
