package com.dragonnest.agent

import android.app.Activity
import android.content.Intent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.animateContentSize
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.Send
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material.icons.outlined.CheckCircle
import androidx.compose.material.icons.outlined.Devices
import androidx.compose.material.icons.outlined.Edit
import androidx.compose.material.icons.outlined.ErrorOutline
import androidx.compose.material.icons.outlined.Hub
import androidx.compose.material.icons.outlined.Lock
import androidx.compose.material.icons.outlined.Person
import androidx.compose.material.icons.outlined.QrCodeScanner
import androidx.compose.material.icons.outlined.SettingsEthernet
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController

private data class PersonaChoice(val id: String, val label: String)

private val personas = listOf(
    PersonaChoice(UserProfile.PERSONA_BALANCED, "Balanced"),
    PersonaChoice(UserProfile.PERSONA_CONCISE, "Concise"),
    PersonaChoice(UserProfile.PERSONA_DETAILED, "Detailed"),
)

@Composable
fun PersonaCareApp(viewModel: PersonaCareViewModel) {
    val navController = rememberNavController()
    NavHost(navController = navController, startDestination = viewModel.initialRoute) {
        composable("connect") {
            ConnectScreen(
                viewModel = viewModel,
                onContinue = {
                    if (viewModel.currentProfile() == null) {
                        navController.navigate("profile") {
                            popUpTo("connect") { inclusive = true }
                        }
                    } else {
                        returnToChat(navController)
                    }
                },
            )
        }
        composable("profile") {
            ProfileScreen(
                viewModel = viewModel,
                canGoBack = navController.previousBackStackEntry != null,
                onBack = navController::popBackStack,
                onSaved = {
                    returnToChat(navController)
                },
            )
        }
        composable("chat") {
            ChatScreen(
                viewModel = viewModel,
                onEditProfile = { navController.navigate("profile") },
                onEditConnection = { navController.navigate("connect") },
            )
        }
    }
}

private fun returnToChat(navController: NavHostController) {
    if (!navController.popBackStack("chat", inclusive = false)) {
        navController.navigate("chat") {
            popUpTo(navController.graph.startDestinationId) { inclusive = true }
        }
    }
}

@Composable
private fun BrandHeader(kicker: String) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(
            text = "PersonaCare",
            style = MaterialTheme.typography.headlineLarge,
            fontWeight = FontWeight.SemiBold,
        )
        Text(
            text = kicker,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            style = MaterialTheme.typography.bodyLarge,
        )
    }
}

@Composable
private fun ConnectScreen(viewModel: PersonaCareViewModel, onContinue: () -> Unit) {
    var host by rememberSaveable { mutableStateOf(viewModel.currentHost()) }
    var port by rememberSaveable { mutableStateOf(viewModel.currentPort()) }
    var code by rememberSaveable { mutableStateOf("") }
    var useTls by rememberSaveable { mutableStateOf(viewModel.currentTls()) }
    var error by rememberSaveable { mutableStateOf("") }
    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartActivityForResult(),
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            runCatching {
                EnrollmentPayload.parse(
                    result.data?.getStringExtra(EnrollmentCaptureActivity.EXTRA_SCAN_RESULT),
                )
            }.onSuccess { payload ->
                viewModel.applyEnrollment(payload)
                onContinue()
            }.onFailure { error = it.message ?: "That enrollment code is not valid" }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .testTag("connect_screen")
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 24.dp, vertical = 36.dp)
            .imePadding(),
        verticalArrangement = Arrangement.spacedBy(20.dp),
    ) {
        BrandHeader("Connect to your DragonNest workspace")
        Spacer(Modifier.height(8.dp))
        OutlinedTextField(
            value = host,
            onValueChange = { host = it; error = "" },
            modifier = Modifier.fillMaxWidth(),
            label = { Text("Server address") },
            leadingIcon = { Icon(Icons.Outlined.Hub, null) },
            singleLine = true,
        )
        OutlinedTextField(
            value = port,
            onValueChange = { port = it.filter(Char::isDigit); error = "" },
            modifier = Modifier.fillMaxWidth(),
            label = { Text("Port") },
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            singleLine = true,
        )
        OutlinedTextField(
            value = code,
            onValueChange = { code = it; error = "" },
            modifier = Modifier.fillMaxWidth(),
            label = { Text("Enrollment code") },
            leadingIcon = { Icon(Icons.Outlined.Lock, null) },
            visualTransformation = PasswordVisualTransformation(),
            supportingText = {
                if (viewModel.hasEnrollment()) Text("Leave blank to keep the current credential")
            },
            singleLine = true,
        )
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Column {
                Text("Secure connection", fontWeight = FontWeight.Medium)
                Text(
                    "TLS",
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            Switch(checked = useTls, onCheckedChange = { useTls = it })
        }
        if (error.isNotBlank()) {
            InlineError(error)
        }
        Button(
            onClick = {
                runCatching { viewModel.saveConnection(host, port, code, useTls) }
                    .onSuccess { onContinue() }
                    .onFailure { error = it.message ?: "Could not save this connection" }
            },
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("Continue")
        }
        OutlinedButton(
            onClick = {
                launcher.launch(EnrollmentCaptureActivity.scanIntent(viewModel.getApplication()))
            },
            modifier = Modifier.fillMaxWidth(),
        ) {
            Icon(Icons.Outlined.QrCodeScanner, null)
            Spacer(Modifier.width(8.dp))
            Text("Scan enrollment QR")
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ProfileScreen(
    viewModel: PersonaCareViewModel,
    canGoBack: Boolean,
    onBack: () -> Unit,
    onSaved: () -> Unit,
) {
    val existing = remember { viewModel.currentProfile() }
    var name by rememberSaveable { mutableStateOf(existing?.personName().orEmpty()) }
    var about by rememberSaveable { mutableStateOf(existing?.profileText().orEmpty()) }
    var persona by rememberSaveable {
        mutableStateOf(existing?.personaId() ?: UserProfile.PERSONA_BALANCED)
    }
    var error by rememberSaveable { mutableStateOf("") }

    Scaffold(
        modifier = Modifier.testTag("profile_screen"),
        topBar = {
            TopAppBar(
                title = { Text("Your profile") },
                navigationIcon = {
                    if (canGoBack) {
                        IconButton(onClick = onBack) {
                            Icon(Icons.AutoMirrored.Outlined.ArrowBack, "Back")
                        }
                    }
                },
            )
        },
    ) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 24.dp, vertical = 12.dp)
                .imePadding(),
            verticalArrangement = Arrangement.spacedBy(20.dp),
        ) {
            BrandHeader("Shape how your AI works with you")
            OutlinedTextField(
                value = name,
                onValueChange = { name = it.take(120); error = "" },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Name") },
                leadingIcon = { Icon(Icons.Outlined.Person, null) },
                singleLine = true,
            )
            OutlinedTextField(
                value = about,
                onValueChange = { about = it.take(500); error = "" },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("About you and your preferences") },
                supportingText = { Text("${about.length}/500") },
                minLines = 6,
                maxLines = 10,
            )
            Text(
                "Default persona",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
            PersonaSelector(selected = persona, onSelected = { persona = it })
            if (error.isNotBlank()) InlineError(error)
            Button(
                onClick = {
                    runCatching { viewModel.saveProfile(name, about, persona) }
                        .onSuccess { onSaved() }
                        .onFailure { error = it.message ?: "Could not save your profile" }
                },
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("Save profile")
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ChatScreen(
    viewModel: PersonaCareViewModel,
    onEditProfile: () -> Unit,
    onEditConnection: () -> Unit,
) {
    val profile = remember { viewModel.currentProfile() }
    val chat by viewModel.chat.collectAsStateWithLifecycle()
    val status by viewModel.agentStatus.collectAsStateWithLifecycle()
    var prompt by rememberSaveable { mutableStateOf("") }
    var persona by rememberSaveable {
        mutableStateOf(profile?.personaId() ?: UserProfile.PERSONA_BALANCED)
    }
    var useProfile by rememberSaveable { mutableStateOf(true) }
    var keepOnPhone by rememberSaveable { mutableStateOf(false) }
    val listState = rememberLazyListState()

    LaunchedEffect(chat.messages.size, chat.sending) {
        val count = chat.messages.size + if (chat.sending) 1 else 0
        if (count > 0) listState.animateScrollToItem(count - 1)
    }

    Scaffold(
        modifier = Modifier.testTag("chat_screen"),
        contentWindowInsets = WindowInsets.navigationBars,
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("PersonaCare", fontWeight = FontWeight.SemiBold)
                        StatusLine(status)
                    }
                },
                actions = {
                    IconButton(onClick = onEditProfile) {
                        Icon(Icons.Outlined.Edit, "Edit profile")
                    }
                    IconButton(onClick = onEditConnection) {
                        Icon(Icons.Outlined.SettingsEthernet, "Connection settings")
                    }
                },
            )
        },
        bottomBar = {
            ChatComposer(
                prompt = prompt,
                onPromptChange = { prompt = it },
                persona = persona,
                onPersonaChange = { persona = it },
                useProfile = useProfile,
                onUseProfileChange = { useProfile = it },
                keepOnPhone = keepOnPhone,
                onKeepOnPhoneChange = { keepOnPhone = it },
                sending = chat.sending,
                canSend = status.state == AgentConnectionState.CONNECTED,
                onSend = {
                    val sent = prompt.trim()
                    if (sent.isNotEmpty()) {
                        viewModel.submit(sent, persona, useProfile, keepOnPhone)
                        prompt = ""
                    }
                },
            )
        },
    ) { padding ->
        if (chat.messages.isEmpty() && !chat.sending) {
            EmptyConversation(
                name = profile?.personName().orEmpty(),
                modifier = Modifier.padding(padding),
            )
        } else {
            LazyColumn(
                state = listState,
                modifier = Modifier
                    .padding(padding)
                    .fillMaxSize(),
                contentPadding = androidx.compose.foundation.layout.PaddingValues(
                    horizontal = 16.dp,
                    vertical = 12.dp,
                ),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                items(chat.messages, key = { it.id }) { message ->
                    MessageBubble(message)
                }
                if (chat.sending) {
                    item("sending") {
                        Row(
                            modifier = Modifier.padding(12.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
                            Spacer(Modifier.width(10.dp))
                            Text("Thinking", color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ChatComposer(
    prompt: String,
    onPromptChange: (String) -> Unit,
    persona: String,
    onPersonaChange: (String) -> Unit,
    useProfile: Boolean,
    onUseProfileChange: (Boolean) -> Unit,
    keepOnPhone: Boolean,
    onKeepOnPhoneChange: (Boolean) -> Unit,
    sending: Boolean,
    canSend: Boolean,
    onSend: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surface)
            .imePadding()
            .navigationBarsPadding()
            .padding(horizontal = 12.dp, vertical = 10.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        PersonaSelector(selected = persona, onSelected = onPersonaChange)
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            CompactToggle("Use profile", useProfile, onUseProfileChange)
            CompactToggle("Keep on phone", keepOnPhone, onKeepOnPhoneChange)
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.Bottom,
        ) {
            OutlinedTextField(
                value = prompt,
                onValueChange = onPromptChange,
                modifier = Modifier.weight(1f),
                placeholder = { Text("Message PersonaCare") },
                minLines = 1,
                maxLines = 4,
                enabled = !sending,
            )
            Spacer(Modifier.width(8.dp))
            FilledIconButton(
                onClick = onSend,
                enabled = canSend && !sending && prompt.isNotBlank(),
                modifier = Modifier.size(52.dp),
            ) {
                Icon(Icons.AutoMirrored.Outlined.Send, "Send")
            }
        }
    }
}

@Composable
private fun PersonaSelector(selected: String, onSelected: (String) -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        personas.forEach { choice ->
            FilterChip(
                selected = selected == choice.id,
                onClick = { onSelected(choice.id) },
                label = { Text(choice.label, maxLines = 1) },
                modifier = Modifier.weight(1f).testTag("persona_${choice.id}"),
            )
        }
    }
}

@Composable
private fun CompactToggle(label: String, checked: Boolean, onChecked: (Boolean) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(label, style = MaterialTheme.typography.labelMedium)
        Spacer(Modifier.width(6.dp))
        Switch(checked = checked, onCheckedChange = onChecked, modifier = Modifier.height(32.dp))
    }
}

@Composable
private fun StatusLine(status: AgentStatus) {
    val color = when (status.state) {
        AgentConnectionState.CONNECTED -> MaterialTheme.colorScheme.primary
        AgentConnectionState.RETRYING -> MaterialTheme.colorScheme.secondary
        else -> MaterialTheme.colorScheme.onSurfaceVariant
    }
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(7.dp).clip(RoundedCornerShape(4.dp)).background(color))
        Spacer(Modifier.width(6.dp))
        Text(
            status.detail.ifBlank { "Connected through DragonNest" },
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            maxLines = 1,
        )
    }
}

@Composable
private fun EmptyConversation(name: String, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier.fillMaxSize().padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Icon(
            Icons.Outlined.Devices,
            null,
            modifier = Modifier.size(44.dp),
            tint = MaterialTheme.colorScheme.primary,
        )
        Spacer(Modifier.height(16.dp))
        Text(
            if (name.isBlank()) "What can I help with?" else "What can I help with, $name?",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.SemiBold,
        )
    }
}

@Composable
private fun MessageBubble(message: ChatMessage) {
    val alignment = if (message.fromUser) Alignment.CenterEnd else Alignment.CenterStart
    val background = when {
        message.failed -> MaterialTheme.colorScheme.secondaryContainer
        message.fromUser -> MaterialTheme.colorScheme.primaryContainer
        else -> MaterialTheme.colorScheme.surfaceVariant
    }
    Box(modifier = Modifier.fillMaxWidth(), contentAlignment = alignment) {
        Column(
            modifier = Modifier
                .fillMaxWidth(0.86f)
                .clip(RoundedCornerShape(8.dp))
                .background(background)
                .padding(14.dp)
                .animateContentSize(),
        ) {
            Text(message.text, lineHeight = 22.sp)
            if (!message.fromUser && message.deviceName.isNotBlank()) {
                Spacer(Modifier.height(8.dp))
                HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(alpha = 0.3f))
                Spacer(Modifier.height(6.dp))
                Text(
                    "Ran on ${message.deviceName}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun InlineError(message: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(Icons.Outlined.ErrorOutline, null, tint = MaterialTheme.colorScheme.error)
        Spacer(Modifier.width(8.dp))
        Text(message, color = MaterialTheme.colorScheme.error)
    }
}
