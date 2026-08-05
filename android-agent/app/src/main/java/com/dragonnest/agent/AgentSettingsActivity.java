package com.dragonnest.agent;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.text.InputType;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import com.dragonnest.proto.SubmitTaskResponse;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** The consumer app flow: connect a device, make it personal, then ask. */
public final class AgentSettingsActivity extends Activity {
    private static final int QR_CAPTURE_REQUEST = 611;
    private AgentConfiguration configuration;
    private EnrollmentStore enrollmentStore;
    private UserProfileStore profileStore;
    private ClientDebugLog debugLog;
    private final ExecutorService queryExecutor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        configuration = new AgentConfiguration(this);
        enrollmentStore = new EnrollmentStore(this);
        profileStore = new UserProfileStore(this);
        debugLog = new ClientDebugLog(this);
        debugLog.add("DragonNest opened");
        if (enrollmentStore.hasCredential() && profileStore.load() != null) {
            showQuery();
        } else {
            showEnrollment();
        }
    }

    @Override
    protected void onDestroy() {
        queryExecutor.shutdownNow();
        super.onDestroy();
    }

    private void showEnrollment() {
        LinearLayout content = page();
        content.addView(title("DragonNest"));
        content.addView(body("Connect this device to start using your personal AI."));

        Button scan = action("Scan enrollment QR");
        scan.setOnClickListener(view -> startQrScan());
        content.addView(scan, matchWidth());
        Button debug = action("Debug");
        debug.setOnClickListener(view -> showDebug());
        content.addView(debug, matchWidth());
        setContentView(scroll(content));
    }

    private void showProfile() {
        renderProfileForm(false);
    }

    private void showSettings() {
        renderProfileForm(true);
    }

    private void renderProfileForm(boolean editing) {
        UserProfile existing = profileStore.load();
        LinearLayout content = page();
        content.addView(title(editing ? "Settings" : "Make it yours"));
        content.addView(body(editing
                ? "Update how DragonNest talks to you. Changes apply right away."
                : "These choices stay with your DragonNest profile."));

        EditText name = field("Your name", existing == null ? "" : existing.personName(),
                InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_CAP_WORDS);
        content.addView(name, matchWidth());

        content.addView(section("What matters most?"));
        RadioGroup mode = choices(
                new String[]{"Balanced", "Fast answers", "Keep it on this device", "Best quality"},
                new String[]{"auto", "fast", "private", "quality"},
                existing == null ? "auto" : existing.preferredMode());
        content.addView(mode, matchWidth());

        content.addView(section("Answer style"));
        RadioGroup style = choices(
                new String[]{"Balanced", "Concise", "Detailed"},
                new String[]{UserProfile.STYLE_BALANCED, UserProfile.STYLE_CONCISE,
                        UserProfile.STYLE_DETAILED},
                existing == null ? UserProfile.STYLE_BALANCED : existing.responseStyle());
        content.addView(style, matchWidth());

        Button continueButton = action(editing ? "Save" : "Continue");
        continueButton.setOnClickListener(view -> {
            try {
                UserProfile profile = new UserProfile(
                        name.getText().toString(),
                        selectedValue(mode),
                        selectedValue(style));
                profileStore.save(profile);
                startAgent();
                if (editing) {
                    Toast.makeText(this, "Settings saved", Toast.LENGTH_SHORT).show();
                }
                showQuery();
            } catch (IllegalArgumentException failure) {
                Toast.makeText(this, failure.getMessage(), Toast.LENGTH_LONG).show();
            }
        });
        content.addView(continueButton, matchWidth());
        if (editing) {
            Button cancel = action("Cancel");
            cancel.setOnClickListener(view -> showQuery());
            content.addView(cancel, matchWidth());
        }
        setContentView(scroll(content));
    }

    private void showQuery() {
        LinearLayout content = page();
        UserProfile profile = profileStore.load();
        content.addView(title("DragonNest"));
        content.addView(body(profile == null
                ? "What would you like help with?"
                : "What would you like help with, " + profile.personName() + "?"));

        EditText prompt = field("Ask anything", "", InputType.TYPE_CLASS_TEXT
                | InputType.TYPE_TEXT_FLAG_MULTI_LINE | InputType.TYPE_TEXT_FLAG_CAP_SENTENCES);
        prompt.setMinLines(4);
        prompt.setGravity(android.view.Gravity.TOP);
        content.addView(prompt, matchWidth());

        TextView result = body("");
        result.setVisibility(View.GONE);
        Button send = action("Send");
        send.setOnClickListener(view -> submitQuery(prompt, result, send));
        content.addView(send, matchWidth());
        Button settings = action("Settings");
        settings.setOnClickListener(view -> showSettings());
        content.addView(settings, matchWidth());
        Button changeRegistration = action("Change registration");
        changeRegistration.setOnClickListener(view -> confirmRegistrationReset());
        content.addView(changeRegistration, matchWidth());
        Button debug = action("Debug");
        debug.setOnClickListener(view -> showDebug());
        content.addView(debug, matchWidth());
        content.addView(result, matchWidth());
        setContentView(scroll(content));
        startAgent();
    }

    private void submitQuery(EditText prompt, TextView result, Button send) {
        String text = prompt.getText().toString().trim();
        if (text.isEmpty()) {
            Toast.makeText(this, "Enter a question first", Toast.LENGTH_SHORT).show();
            return;
        }
        send.setEnabled(false);
        debugLog.add("Submitting a user query");
        result.setText("Thinking...");
        result.setVisibility(View.VISIBLE);
        queryExecutor.execute(() -> {
            try {
                SubmitTaskResponse response = new BrainTaskClient(configuration).submit(text);
                runOnUiThread(() -> {
                    result.setText(response.getSuccess()
                            ? response.getOutputText()
                            : friendlyError(response));
                    send.setEnabled(true);
                    debugLog.add(response.getSuccess()
                            ? "Query completed" : "Query failed: " + response.getErrorCode());
                });
            } catch (Exception failure) {
                runOnUiThread(() -> {
                    result.setText("DragonNest could not reach your workspace. Please try again.");
                    send.setEnabled(true);
                    debugLog.add("Query transport failed: " + failure.getClass().getSimpleName());
                });
            }
        });
    }

    private static String friendlyError(SubmitTaskResponse response) {
        if (response.getErrorCode().equals("NO_ELIGIBLE_FALLBACK")) {
            return "Your device is getting ready. Please try again in a few seconds.";
        }
        if (response.getErrorCode().equals("STEERING_UNAVAILABLE")) {
            return "That answer style is not available for the selected model yet.";
        }
        return "DragonNest could not complete that request. Please try again.";
    }

    private void startAgent() {
        requestNotificationPermission();
        Intent start = new Intent(this, AgentForegroundService.class);
        start.setAction(AgentForegroundService.ACTION_RELOAD);
        startForegroundService(start);
    }

    private void startQrScan() {
        debugLog.add("Opening QR camera scanner");
        startActivityForResult(
                EnrollmentCaptureActivity.scanIntent(this), QR_CAPTURE_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode != QR_CAPTURE_REQUEST) {
            super.onActivityResult(requestCode, resultCode, data);
            return;
        }
        if (resultCode != RESULT_OK || data == null) {
            debugLog.add("QR scanner closed without a result");
            return;
        }
        try {
            confirmEnrollment(EnrollmentPayload.parse(
                    data.getStringExtra(EnrollmentCaptureActivity.EXTRA_SCAN_RESULT)));
            debugLog.add("Enrollment QR read successfully");
        } catch (Exception failure) {
            debugLog.add("Enrollment QR rejected: " + failure.getMessage());
            Toast.makeText(this, failure.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private void confirmEnrollment(EnrollmentPayload payload) {
        new AlertDialog.Builder(this)
                .setTitle("Connect this device")
                .setMessage("This securely connects your device to DragonNest.")
                .setNegativeButton("Cancel", null)
                .setPositiveButton("Continue", (dialog, which) -> applyEnrollment(payload))
                .show();
    }

    private void applyEnrollment(EnrollmentPayload payload) {
        try {
            configuration.saveEnrollmentEndpoint(
                    payload.brainHost(), payload.brainPort(), payload.useTls());
            enrollmentStore.save(payload.credential());
            debugLog.add("Enrollment saved locally; collecting profile");
            showProfile();
        } catch (Exception failure) {
            debugLog.add("Enrollment could not be saved: " + failure.getClass().getSimpleName());
            Toast.makeText(this, "Could not save enrollment", Toast.LENGTH_LONG).show();
        }
    }

    private void requestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU
                && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 100);
        }
    }

    private void confirmRegistrationReset() {
        new AlertDialog.Builder(this)
                .setTitle("Change registration")
                .setMessage("This stops the device agent and clears this device's local enrollment and profile.")
                .setNegativeButton("Cancel", null)
                .setPositiveButton("Reset", (dialog, which) -> resetRegistration())
                .show();
    }

    private void resetRegistration() {
        stopService(new Intent(this, AgentForegroundService.class));
        enrollmentStore.clear();
        profileStore.clear();
        configuration.clearEnrollmentEndpoint();
        debugLog.add("Local registration reset");
        showEnrollment();
    }

    private void showDebug() {
        LinearLayout content = page();
        content.addView(title("Client debug"));
        TextView events = new TextView(this);
        events.setText(debugLog.read());
        events.setTextSize(13);
        content.addView(events, matchWidth());
        Button refresh = action("Refresh");
        refresh.setOnClickListener(view -> events.setText(debugLog.read()));
        content.addView(refresh, matchWidth());
        Button clear = action("Clear debug");
        clear.setOnClickListener(view -> {
            debugLog.clear();
            events.setText(debugLog.read());
        });
        content.addView(clear, matchWidth());
        Button back = action("Back");
        back.setOnClickListener(view -> {
            if (enrollmentStore.hasCredential() && profileStore.load() != null) {
                showQuery();
            } else {
                showEnrollment();
            }
        });
        content.addView(back, matchWidth());
        setContentView(scroll(content));
    }

    private LinearLayout page() {
        int padding = (int) (20 * getResources().getDisplayMetrics().density);
        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        content.setPadding(padding, padding, padding, padding);
        return content;
    }

    private TextView title(String text) {
        TextView view = new TextView(this);
        view.setText(text);
        view.setTextSize(28);
        view.setPadding(0, 0, 0, 12);
        return view;
    }

    private TextView body(String text) {
        TextView view = new TextView(this);
        view.setText(text);
        view.setTextSize(17);
        view.setPadding(0, 0, 0, 16);
        return view;
    }

    private TextView section(String text) {
        TextView view = new TextView(this);
        view.setText(text);
        view.setTextSize(18);
        view.setPadding(0, 16, 0, 4);
        return view;
    }

    private EditText field(String hint, String value, int inputType) {
        EditText view = new EditText(this);
        view.setHint(hint);
        view.setText(value);
        view.setInputType(inputType);
        view.setPadding(0, 8, 0, 8);
        return view;
    }

    private RadioGroup choices(String[] labels, String[] values, String selected) {
        RadioGroup group = new RadioGroup(this);
        for (int index = 0; index < labels.length; index++) {
            RadioButton button = new RadioButton(this);
            button.setId(View.generateViewId());
            button.setText(labels[index]);
            button.setTag(values[index]);
            button.setChecked(values[index].equals(selected));
            group.addView(button, matchWidth());
        }
        return group;
    }

    private static String selectedValue(RadioGroup group) {
        RadioButton selected = group.findViewById(group.getCheckedRadioButtonId());
        if (selected == null) {
            throw new IllegalArgumentException("Make a selection to continue");
        }
        return String.valueOf(selected.getTag());
    }

    private Button action(String text) {
        Button button = new Button(this);
        button.setText(text);
        return button;
    }

    private ScrollView scroll(LinearLayout content) {
        ScrollView scroll = new ScrollView(this);
        scroll.addView(content);
        return scroll;
    }

    private static ViewGroup.LayoutParams matchWidth() {
        return new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
    }
}
