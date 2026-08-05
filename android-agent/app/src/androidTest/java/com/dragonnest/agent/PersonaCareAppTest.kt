package com.dragonnest.agent

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import org.junit.Rule
import org.junit.Test

class PersonaCareAppTest {
    @get:Rule
    val compose = createAndroidComposeRule<AgentSettingsActivity>()

    @Test
    fun freshInstallShowsConnectionWorkflow() {
        compose.onNodeWithTag("connect_screen").assertIsDisplayed()
        compose.onNodeWithText("PersonaCare").assertIsDisplayed()
        compose.onNodeWithText("Server address").assertIsDisplayed()
        compose.onNodeWithText("Scan enrollment QR").assertIsDisplayed()
    }
}
