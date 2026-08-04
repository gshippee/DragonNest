package com.dragonnest.agent;

import android.content.Context;
import android.os.Build;
import android.os.Environment;
import android.os.StatFs;

import com.dragonnest.proto.HardwareInventory;

import java.util.Arrays;

/** Collects static platform attributes without claiming a vendor runtime is installed. */
public final class AndroidHardwareInventory {
    private final Context context;

    public AndroidHardwareInventory(Context context) {
        this.context = context.getApplicationContext();
    }

    public HardwareInventory snapshot() {
        StatFs data = new StatFs(Environment.getDataDirectory().getAbsolutePath());
        String socManufacturer = "";
        String socModel = "";
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            socManufacturer = Build.SOC_MANUFACTURER;
            socModel = Build.SOC_MODEL;
        }
        return HardwareInventory.newBuilder()
                .setManufacturer(Build.MANUFACTURER)
                .setModel(Build.MODEL)
                .setDevice(Build.DEVICE)
                .setOsVersion(Build.VERSION.RELEASE)
                .setApiLevel(Build.VERSION.SDK_INT)
                .setSocManufacturer(socManufacturer)
                .setSocModel(socModel)
                .addAllCpuAbis(Arrays.asList(Build.SUPPORTED_ABIS))
                .setCpuCoreCount(Runtime.getRuntime().availableProcessors())
                .setTotalStorageMb(data.getTotalBytes() / (1024L * 1024L))
                .setAvailableStorageMb(data.getAvailableBytes() / (1024L * 1024L))
                .setNpuStatus("not_probed")
                .build();
    }
}
