package com.dragonnest.geniexsteeringlab;

import android.annotation.SuppressLint;
import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.ArrayAdapter;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.json.JSONObject;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * SteerLab — diagnostic front-end for GenieX runtime activation steering.
 *
 * The panel is deliberately verbose: every request logs which graph phase
 * (prefill vs decode) received the aux tensors, straight from the forked
 * runtime's geniex_llm_get_aux_stats counters — not inferred client-side.
 */
public class MainActivity extends AppCompatActivity {

    private static final String VEC_ASSET = "steering_vector_layer7_unit.bin";
    private static final int HIDDEN = 1024;
    // Deployed bundle's genie_config.json declares context.size = 512 (prompt +
    // generation KV budget); leave headroom below that instead of hard-capping
    // generation at an arbitrary demo value.
    private static final int MAX_GEN_TOKENS = 480;

    private final ExecutorService exec = Executors.newSingleThreadExecutor();

    private TextView modelStatus;
    private TextView alphaLabel;
    private SeekBar alphaSeek;
    private Spinner vectorSpinner;
    private EditText promptEdit;
    private Button generateBtn;
    private TextView outputView;
    private TextView diagView;
    private ScrollView diagScroll;

    private float[] layer7Vector;   // unit L2 norm, length 1024
    private float[] randomVector;   // control, same norm
    private boolean modelReady = false;
    private int requestNo = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        modelStatus = findViewById(R.id.model_status);
        alphaLabel = findViewById(R.id.alpha_label);
        alphaSeek = findViewById(R.id.alpha_seek);
        vectorSpinner = findViewById(R.id.vector_spinner);
        promptEdit = findViewById(R.id.prompt_edit);
        generateBtn = findViewById(R.id.generate_btn);
        outputView = findViewById(R.id.output_view);
        diagView = findViewById(R.id.diag_view);
        diagScroll = findViewById(R.id.diag_scroll);

        vectorSpinner.setAdapter(new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item,
                new String[]{"layer-7 verbosity (real)", "random control (same norm)", "unsteered (no aux inputs)"}));

        // Slider: -10.0 .. +10.0 in 0.5 steps -> 41 positions, 20 == 0.
        alphaSeek.setMax(40);
        alphaSeek.setProgress(20);
        alphaSeek.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar b, int p, boolean u) { updateAlphaLabel(); }
            @Override public void onStartTrackingTouch(SeekBar b) {}
            @Override public void onStopTrackingTouch(SeekBar b) {}
        });
        updateAlphaLabel();

        loadVectors();

        generateBtn.setEnabled(false);
        generateBtn.setOnClickListener(v -> runGenerate());

        logDiag("SteerLab " + BuildConfigInfo() + " — forked GenieX aux-input runtime");
        initModel();
    }

    private String BuildConfigInfo() {
        return "0.1 (" + android.os.Build.MODEL + ", " + android.os.Build.SOC_MODEL + ")";
    }

    private float alphaValue() {
        return (alphaSeek.getProgress() - 20) / 2.0f;
    }

    @SuppressLint("SetTextI18n")
    private void updateAlphaLabel() {
        alphaLabel.setText(String.format(Locale.US, "alpha = %+.1f", alphaValue()));
    }

    private void loadVectors() {
        try {
            byte[] raw;
            try (DataInputStream in = new DataInputStream(getAssets().open(VEC_ASSET))) {
                raw = new byte[HIDDEN * 4];
                in.readFully(raw);
            }
            ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
            layer7Vector = new float[HIDDEN];
            for (int i = 0; i < HIDDEN; i++) layer7Vector[i] = bb.getFloat();

            Random rng = new Random(20260807L);
            randomVector = new float[HIDDEN];
            double norm = 0;
            for (int i = 0; i < HIDDEN; i++) { randomVector[i] = (float) rng.nextGaussian(); norm += randomVector[i] * randomVector[i]; }
            float inv = (float) (1.0 / Math.sqrt(norm));
            for (int i = 0; i < HIDDEN; i++) randomVector[i] *= inv;
            logDiag("vectors: layer-7 unit vector loaded (" + HIDDEN + " floats), random control seeded");
        } catch (Exception e) {
            logDiag("ERROR loading vector asset: " + e);
        }
    }

    private File bundleDir() {
        // Prefer internal app storage (ext4 — QNN context loading is happiest
        // there); fall back to app-private external storage, which adb can
        // push to directly:
        //   adb push <bundle>/ /sdcard/Android/data/com.dragonnest.geniexsteeringlab/files/qwen-bundle/
        // and copy internal via: adb shell run-as com.dragonnest.geniexsteeringlab \
        //   cp -r /sdcard/Android/data/com.dragonnest.geniexsteeringlab/files/qwen-bundle files/
        File internal = new File(getFilesDir(), "qwen-bundle");
        if (new File(internal, "genie_config.json").isFile()) return internal;
        return new File(getExternalFilesDir(null), "qwen-bundle");
    }

    private void initModel() {
        File bundle = bundleDir();
        modelStatus.setText("loading " + bundle.getAbsolutePath() + " …");
        exec.submit(() -> {
            try {
            String libDir = getApplicationInfo().nativeLibraryDir;
            logDiagAsync("init: GENIEX_PLUGIN_PATH=" + libDir);
            logDiagAsync("init: bundle=" + bundle.getAbsolutePath() + (bundle.isDirectory() ? "" : "  [MISSING]"));
            String result = GenieXBridge.init(libDir, bundle.getAbsolutePath());
            runOnUiThread(() -> {
                try {
                    JSONObject o = new JSONObject(result);
                    if (o.optBoolean("ok")) {
                        modelReady = true;
                        generateBtn.setEnabled(true);
                        JSONObject s = o.optJSONObject("stats");
                        modelStatus.setText(String.format(Locale.US,
                                "model READY — load %d ms, context loads: %d (reused hereafter)",
                                o.optLong("load_ms"), s != null ? s.optLong("context_loads") : 1));
                        logDiag("init OK: " + result);
                    } else {
                        modelStatus.setText("LOAD FAILED — see diagnostics");
                        logDiag("init FAILED: " + result);
                    }
                } catch (Exception e) {
                    logDiag("init parse error: " + e + " raw=" + result);
                }
            });
            } catch (Throwable t) {
                logDiagAsync("init CRASHED: " + t);
                runOnUiThread(() -> modelStatus.setText("LOAD CRASHED — see diagnostics"));
            }
        });
    }

    private void runGenerate() {
        if (!modelReady) return;
        final float alpha = alphaValue();
        final int vecSel = vectorSpinner.getSelectedItemPosition();
        final boolean useAux = vecSel != 2;
        final float[] vec = vecSel == 0 ? layer7Vector : (vecSel == 1 ? randomVector : null);
        final String vecName = vecSel == 0 ? "layer7-verbosity" : (vecSel == 1 ? "random-control" : "none");
        final String prompt = promptEdit.getText().toString();
        final int req = ++requestNo;

        generateBtn.setEnabled(false);
        outputView.setText("generating …");
        logDiag(String.format(Locale.US,
                "req #%d ▸ submit  aux=%s  alpha=%+.1f  vector=%s  (same loaded context, no reload)",
                req, useAux ? "ON" : "OFF", alpha, vecName));

        exec.submit(() -> {
            try {
            String result = GenieXBridge.generate(prompt, alpha, vec, useAux, MAX_GEN_TOKENS);
            runOnUiThread(() -> {
                generateBtn.setEnabled(true);
                try {
                    JSONObject o = new JSONObject(result);
                    if (o.optBoolean("ok")) {
                        outputView.setText(o.optString("text"));
                        JSONObject s = o.optJSONObject("stats");
                        logDiag(String.format(Locale.US,
                                "req #%d ◂ done in %d ms  (%d prompt tok, %d gen tok, %.1f tok/s)",
                                req, o.optLong("ms"), o.optLong("prompt_tokens"),
                                o.optLong("generated_tokens"), o.optDouble("tps")));
                        logDiag(String.format(Locale.US,
                                "req #%d ◂ PREFILL graphs got %d aux writes, DECODE graphs got %d aux writes",
                                req, o.optLong("delta_prefill_writes"), o.optLong("delta_decode_writes")));
                        if (s != null) {
                            logDiag(String.format(Locale.US,
                                    "totals ▸ prefill_writes=%d decode_writes=%d aux_requests=%d/%d context_loads=%d",
                                    s.optLong("prefill_writes"), s.optLong("decode_writes"),
                                    s.optLong("aux_requests"), s.optLong("total_requests"),
                                    s.optLong("context_loads")));
                        }
                    } else {
                        outputView.setText("(request failed — see diagnostics)");
                        logDiag("req #" + req + " FAILED: " + result);
                    }
                } catch (Exception e) {
                    logDiag("req #" + req + " parse error: " + e + " raw=" + result);
                }
            });
            } catch (Throwable t) {
                logDiagAsync("req #" + req + " CRASHED: " + t);
                runOnUiThread(() -> generateBtn.setEnabled(true));
            }
        });
    }

    private void logDiag(String line) {
        diagView.append(line + "\n");
        diagScroll.post(() -> diagScroll.fullScroll(ScrollView.FOCUS_DOWN));
    }

    private void logDiagAsync(String line) {
        runOnUiThread(() -> logDiag(line));
    }
}
