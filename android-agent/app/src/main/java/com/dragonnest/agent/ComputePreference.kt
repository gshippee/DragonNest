package com.dragonnest.agent

enum class ComputePreference(
    val wireValue: String,
    val displayName: String,
    val description: String,
) {
    AUTO("auto", "Auto", "DragonNest chooses"),
    LOCAL("local", "Local", "Stay on this device"),
    ELASTIC("elastic", "Elastic", "Use available devices together"),
    QUALITY("quality", "Quality", "Prefer the strongest model"),
    ;

    companion object {
        fun fromWireValue(value: String): ComputePreference =
            entries.firstOrNull { it.wireValue == value } ?: AUTO
    }
}
