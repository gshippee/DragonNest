package com.dragonnest.agent;

import android.content.Context;
import android.content.SharedPreferences;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/** Small on-device diagnostic log. Never records enrollment credentials. */
public final class ClientDebugLog {
    private static final String PREFERENCES = "client-debug";
    private static final String EVENTS = "events";
    private static final int MAX_CHARACTERS = 8_000;
    private final SharedPreferences preferences;

    public ClientDebugLog(Context context) {
        preferences = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE);
    }

    public synchronized void add(String message) {
        String timestamp = new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date());
        String current = preferences.getString(EVENTS, "");
        String next = timestamp + "  " + sanitize(message) + "\n" + current;
        if (next.length() > MAX_CHARACTERS) {
            next = next.substring(0, MAX_CHARACTERS);
        }
        preferences.edit().putString(EVENTS, next).commit();
    }

    public synchronized String read() {
        String events = preferences.getString(EVENTS, "");
        return events == null || events.isBlank() ? "No client events yet." : events;
    }

    public synchronized void clear() {
        preferences.edit().remove(EVENTS).commit();
    }

    private static String sanitize(String message) {
        return message == null ? "" : message.replace('\n', ' ').trim();
    }
}
