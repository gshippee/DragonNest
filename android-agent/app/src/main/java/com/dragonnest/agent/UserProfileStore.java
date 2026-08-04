package com.dragonnest.agent;

import android.content.Context;
import android.content.SharedPreferences;

/** Persists the person-facing choices on the client device. */
public final class UserProfileStore {
    private static final String PREFERENCES = "user-profile";
    private final SharedPreferences preferences;

    public UserProfileStore(Context context) {
        preferences = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE);
    }

    public UserProfile load() {
        String name = preferences.getString("person_name", "");
        if (name == null || name.trim().isEmpty()) {
            return null;
        }
        return new UserProfile(
                name,
                preferences.getString("preferred_mode", "auto"),
                preferences.getString("response_style", UserProfile.STYLE_BALANCED));
    }

    public void save(UserProfile profile) {
        preferences.edit()
                .putString("person_name", profile.personName())
                .putString("preferred_mode", profile.preferredMode())
                .putString("response_style", profile.responseStyle())
                .apply();
    }

    public void clear() {
        preferences.edit().clear().apply();
    }
}
