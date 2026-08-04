package com.dragonnest.agent;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.google.zxing.ResultPoint;
import com.journeyapps.barcodescanner.BarcodeCallback;
import com.journeyapps.barcodescanner.BarcodeResult;
import com.journeyapps.barcodescanner.DecoratedBarcodeView;

import java.util.List;

/** In-app QR camera with explicit Android runtime permission ownership. */
public final class EnrollmentCaptureActivity extends Activity {
    public static final String EXTRA_SCAN_RESULT = "SCAN_RESULT";
    private static final int CAMERA_PERMISSION_REQUEST = 612;
    private static final String SCAN_ACTION = "com.google.zxing.client.android.SCAN";
    private static final String SCAN_FORMATS = "SCAN_FORMATS";
    private static final String QR_CODE = "QR_CODE";
    private static final String PROMPT_MESSAGE = "PROMPT_MESSAGE";
    private static final String BEEP_ENABLED = "BEEP_ENABLED";
    private static final String ORIENTATION_LOCKED = "SCAN_ORIENTATION_LOCKED";

    private DecoratedBarcodeView barcodeView;
    private boolean resumed;
    private boolean scannerActive;

    public static Intent scanIntent(Context context) {
        return new Intent(context, EnrollmentCaptureActivity.class)
                .setAction(SCAN_ACTION)
                .putExtra(SCAN_FORMATS, QR_CODE)
                .putExtra(PROMPT_MESSAGE, "Scan DragonNest enrollment")
                .putExtra(BEEP_ENABLED, false)
                .putExtra(ORIENTATION_LOCKED, false);
    }

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        if (!getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY)) {
            showCameraUnavailable();
            return;
        }
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startScanner();
        } else {
            showCameraPermission();
            requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST);
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        resumed = true;
        if (scannerActive && barcodeView != null) {
            barcodeView.resume();
        }
    }

    @Override
    protected void onPause() {
        resumed = false;
        if (scannerActive && barcodeView != null) {
            barcodeView.pause();
        }
        super.onPause();
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode != CAMERA_PERMISSION_REQUEST) {
            return;
        }
        if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startScanner();
        } else {
            showCameraPermission();
        }
    }

    private void startScanner() {
        barcodeView = new DecoratedBarcodeView(this);
        barcodeView.initializeFromIntent(getIntent());
        barcodeView.decodeSingle(new BarcodeCallback() {
            @Override
            public void barcodeResult(BarcodeResult result) {
                if (!scannerActive) {
                    return;
                }
                scannerActive = false;
                barcodeView.pause();
                setResult(RESULT_OK, new Intent().putExtra(EXTRA_SCAN_RESULT, result.getText()));
                finish();
            }

            @Override
            public void possibleResultPoints(List<ResultPoint> resultPoints) {
                // No-op: the embedded view draws candidate points itself.
            }
        });
        scannerActive = true;
        setContentView(barcodeView);
        if (resumed) {
            barcodeView.resume();
        }
    }

    private void showCameraPermission() {
        showMessage(
                "Camera access needed",
                "Allow camera access to scan your DragonNest enrollment code.",
                "Allow camera",
                () -> requestPermissions(
                        new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST));
    }

    private void showCameraUnavailable() {
        showMessage(
                "Camera unavailable",
                "This device does not currently provide a camera for QR enrollment.",
                "Close",
                this::finish);
    }

    private void showMessage(String title, String detail, String action, Runnable onClick) {
        int padding = (int) (24 * getResources().getDisplayMetrics().density);
        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        content.setPadding(padding, padding, padding, padding);
        TextView heading = new TextView(this);
        heading.setText(title);
        heading.setTextSize(24);
        TextView body = new TextView(this);
        body.setText(detail);
        body.setTextSize(17);
        Button button = new Button(this);
        button.setText(action);
        button.setOnClickListener(view -> onClick.run());
        content.addView(heading, matchWidth());
        content.addView(body, matchWidth());
        content.addView(button, matchWidth());
        setContentView(content);
    }

    private static ViewGroup.LayoutParams matchWidth() {
        return new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
    }
}
