package com.dragonnest.agent

import org.junit.Assert.assertEquals
import org.junit.Test

class ComputePreferenceTest {
    @Test
    fun productPreferencesMapToStableWireValues() {
        assertEquals("auto", ComputePreference.AUTO.wireValue)
        assertEquals("local", ComputePreference.LOCAL.wireValue)
        assertEquals("elastic", ComputePreference.ELASTIC.wireValue)
        assertEquals("quality", ComputePreference.QUALITY.wireValue)
    }

    @Test
    fun unknownPersistedPreferenceFallsBackToAuto() {
        assertEquals(ComputePreference.AUTO, ComputePreference.fromWireValue("private"))
    }
}
