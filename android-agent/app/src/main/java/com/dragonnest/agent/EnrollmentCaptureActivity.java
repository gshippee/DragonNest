package com.dragonnest.agent;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;

import com.journeyapps.barcodescanner.CaptureManager;
import com.journeyapps.barcodescanner.DecoratedBarcodeView;

/**
 * An in-app camera scanner. It deliberately does not use an ACTION_SCAN
 * intent, so Android cannot hand scanning off to a document or photo picker.
 */
public final class EnrollmentCaptureActivity extends Activity {
    public static final String EXTRA_SCAN_RESULT = "SCAN_RESULT";
    private static final String SCAN_ACTION = "com.google.zxing.client.android.SCAN";
    private static final String SCAN_FORMATS = "SCAN_FORMATS";
    private static final String QR_CODE = "QR_CODE";
    private static final String PROMPT_MESSAGE = "PROMPT_MESSAGE";
    private static final String BEEP_ENABLED = "BEEP_ENABLED";
    private static final String ORIENTATION_LOCKED = "SCAN_ORIENTATION_LOCKED";
    private CaptureManager captureManager;

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
        DecoratedBarcodeView barcodeView = new DecoratedBarcodeView(this);
        setContentView(barcodeView);
        captureManager = new CaptureManager(this, barcodeView);
        captureManager.initializeFromIntent(getIntent(), state);
        captureManager.decode();
    }

    @Override
    protected void onResume() {
        super.onResume();
        captureManager.onResume();
    }

    @Override
    protected void onPause() {
        captureManager.onPause();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        captureManager.onDestroy();
        super.onDestroy();
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        captureManager.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    @Override
    protected void onSaveInstanceState(Bundle state) {
        captureManager.onSaveInstanceState(state);
        super.onSaveInstanceState(state);
    }
}
