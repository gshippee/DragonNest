package com.dragonnest.agent;

import android.app.Activity;
import android.os.Bundle;

import com.journeyapps.barcodescanner.CaptureManager;
import com.journeyapps.barcodescanner.DecoratedBarcodeView;

/**
 * An in-app camera scanner. It deliberately does not use an ACTION_SCAN
 * intent, so Android cannot hand scanning off to a document or photo picker.
 */
public final class EnrollmentCaptureActivity extends Activity {
    public static final String EXTRA_SCAN_RESULT = "SCAN_RESULT";
    private CaptureManager captureManager;

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
