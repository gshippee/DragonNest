package com.dragonnest.agent;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.text.InputType;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.zxing.integration.android.IntentIntegrator;
import com.google.zxing.integration.android.IntentResult;

public final class AgentSettingsActivity extends Activity {
    private static final int CAMERA_PERMISSION_REQUEST = 101;
    private AgentConfiguration configuration;
    private EnrollmentStore enrollmentStore;
    private EditText host;
    private EditText port;
    private EditText displayName;
    private EditText enrollmentToken;
    private EditText simulatedBattery;
    private EditText simulatedThermal;
    private EditText simulatedCpu;
    private EditText simulatedAccelerator;
    private EditText simulatedRtt;
    private CheckBox tls;
    private CheckBox simulatedOffline;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        configuration = new AgentConfiguration(this);
        enrollmentStore = new EnrollmentStore(this);
        setContentView(buildContent());
        requestNotificationPermission();
    }

    private ScrollView buildContent() {
        int padding = (int) (20 * getResources().getDisplayMetrics().density);
        LinearLayout form = new LinearLayout(this);
        form.setOrientation(LinearLayout.VERTICAL);
        form.setPadding(padding, padding, padding, padding);

        TextView title = new TextView(this);
        title.setText("DragonNest Agent");
        title.setTextSize(26);
        form.addView(title, matchWidth());

        TextView deviceId = new TextView(this);
        deviceId.setText("Device ID: " + configuration.deviceId());
        form.addView(deviceId, matchWidth());

        TextView hardware = new TextView(this);
        AndroidRuntimeCatalog runtimeCatalog = AndroidRuntimeCatalog.create(this);
        var hardwareInfo = new AndroidHardwareInventory(this, runtimeCatalog).snapshot();
        hardware.setText("Hardware: " + hardwareInfo.getModel()
                + " · " + hardwareInfo.getSocModel()
                + " · NPU " + hardwareInfo.getNpuStatus());
        form.addView(hardware, matchWidth());

        Button scan = new Button(this);
        scan.setText("Scan enrollment QR");
        scan.setOnClickListener(view -> startQrScan());
        form.addView(scan, matchWidth());

        host = field("Brain host", configuration.brainHost(), InputType.TYPE_CLASS_TEXT);
        port = field(
                "Brain gRPC port",
                String.valueOf(configuration.brainPort()),
                InputType.TYPE_CLASS_NUMBER);
        displayName = field(
                "Display name", configuration.displayName(), InputType.TYPE_CLASS_TEXT);
        enrollmentToken = field(
                "Manual enrollment token", loadEnrollmentToken(),
                InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_PASSWORD);
        form.addView(host, matchWidth());
        form.addView(port, matchWidth());
        form.addView(displayName, matchWidth());
        form.addView(enrollmentToken, matchWidth());

        tls = new CheckBox(this);
        tls.setText("Use TLS");
        tls.setChecked(configuration.useTls());
        form.addView(tls, matchWidth());

        SimulationState simulation = configuration.simulation();
        TextView simulationTitle = new TextView(this);
        simulationTitle.setText("Simulation overrides (blank uses platform telemetry)");
        form.addView(simulationTitle, matchWidth());
        simulatedBattery = numericField("Battery percentage", simulation.batteryPercentage());
        simulatedThermal = numericField("Thermal level 0..1", simulation.thermalLevel());
        simulatedCpu = numericField("CPU utilization 0..1", simulation.cpuUtilization());
        simulatedAccelerator = numericField(
                "Accelerator utilization 0..1", simulation.acceleratorUtilization());
        simulatedRtt = numericField("Network RTT ms", simulation.networkRttMs());
        form.addView(simulatedBattery, matchWidth());
        form.addView(simulatedThermal, matchWidth());
        form.addView(simulatedCpu, matchWidth());
        form.addView(simulatedAccelerator, matchWidth());
        form.addView(simulatedRtt, matchWidth());
        simulatedOffline = new CheckBox(this);
        simulatedOffline.setText("Simulated offline");
        simulatedOffline.setChecked(simulation.offline());
        form.addView(simulatedOffline, matchWidth());

        Button start = new Button(this);
        start.setText("Save and start agent");
        start.setOnClickListener(view -> saveAndStart());
        form.addView(start, matchWidth());

        Button stop = new Button(this);
        stop.setText("Stop agent");
        stop.setOnClickListener(view -> {
            stopService(new Intent(this, AgentForegroundService.class));
            Toast.makeText(this, "Agent stopped", Toast.LENGTH_SHORT).show();
        });
        form.addView(stop, matchWidth());

        ScrollView scroll = new ScrollView(this);
        scroll.addView(form);
        return scroll;
    }

    private EditText field(String hint, String value, int inputType) {
        EditText field = new EditText(this);
        field.setHint(hint);
        field.setText(value);
        field.setInputType(inputType);
        field.setSingleLine(true);
        return field;
    }

    private EditText numericField(String hint, Float value) {
        return field(
                hint,
                value == null ? "" : String.valueOf(value),
                InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_DECIMAL);
    }

    private void saveAndStart() {
        try {
            int parsedPort = Integer.parseInt(port.getText().toString().trim());
            if (parsedPort < 1 || parsedPort > 65535) {
                throw new IllegalArgumentException("Port must be between 1 and 65535");
            }
            configuration.save(
                    host.getText().toString(),
                    parsedPort,
                    tls.isChecked(),
                    displayName.getText().toString());
            String manualCredential = enrollmentToken.getText().toString().trim();
            if (!manualCredential.isEmpty()) {
                enrollmentStore.save(manualCredential);
            }
            configuration.saveSimulation(
                    optionalFloat(simulatedBattery),
                    boundedFloat(simulatedThermal, 0, 1),
                    boundedFloat(simulatedCpu, 0, 1),
                    boundedFloat(simulatedAccelerator, 0, 1),
                    optionalFloat(simulatedRtt),
                    simulatedOffline.isChecked());
            Intent start = new Intent(this, AgentForegroundService.class);
            start.setAction(AgentForegroundService.ACTION_RELOAD);
            startForegroundService(start);
            Toast.makeText(this, "Agent started", Toast.LENGTH_SHORT).show();
        } catch (Exception failure) {
            Toast.makeText(this, failure.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private void startQrScan() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M
                && checkSelfPermission(Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(
                    new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST);
            return;
        }
        launchQrScanner();
    }

    private void launchQrScanner() {
        new IntentIntegrator(this)
                .setDesiredBarcodeFormats(IntentIntegrator.QR_CODE)
                .setPrompt("Scan DragonNest enrollment")
                .setBeepEnabled(false)
                .setOrientationLocked(false)
                .initiateScan();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        IntentResult result = IntentIntegrator.parseActivityResult(
                requestCode, resultCode, data);
        if (result == null) {
            super.onActivityResult(requestCode, resultCode, data);
            return;
        }
        if (result.getContents() == null) {
            return;
        }
        try {
            confirmEnrollment(EnrollmentPayload.parse(result.getContents()));
        } catch (Exception failure) {
            Toast.makeText(this, failure.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private void confirmEnrollment(EnrollmentPayload payload) {
        String address = payload.brainHost() + ":" + payload.brainPort()
                + (payload.useTls() ? " · TLS" : "");
        new AlertDialog.Builder(this)
                .setTitle("Enroll with DragonNest")
                .setMessage(address)
                .setNegativeButton("Cancel", null)
                .setPositiveButton("Enroll", (dialog, which) -> applyEnrollment(payload))
                .show();
    }

    private void applyEnrollment(EnrollmentPayload payload) {
        try {
            host.setText(payload.brainHost());
            port.setText(String.valueOf(payload.brainPort()));
            tls.setChecked(payload.useTls());
            enrollmentStore.save(payload.credential());
            enrollmentToken.setText("");
            saveAndStart();
        } catch (Exception failure) {
            Toast.makeText(this, failure.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST
                && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            launchQrScanner();
        }
    }

    private static Float optionalFloat(EditText field) {
        String value = field.getText().toString().trim();
        return value.isEmpty() ? null : Float.parseFloat(value);
    }

    private static Float boundedFloat(EditText field, float minimum, float maximum) {
        Float value = optionalFloat(field);
        if (value != null && (value < minimum || value > maximum)) {
            throw new IllegalArgumentException(
                    field.getHint() + " must be between " + minimum + " and " + maximum);
        }
        return value;
    }

    private String loadEnrollmentToken() {
        try {
            String stored = enrollmentStore.load();
            return stored.isEmpty() ? "dev-token" : stored;
        } catch (Exception failure) {
            return "";
        }
    }

    private void requestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU
                && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 100);
        }
    }

    private static ViewGroup.LayoutParams matchWidth() {
        return new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
    }
}
