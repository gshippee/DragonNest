package com.dragonnest.agent;

import android.content.Context;
import android.content.SharedPreferences;
import android.security.keystore.KeyGenParameterSpec;
import android.security.keystore.KeyProperties;
import android.util.Base64;

import org.json.JSONObject;

import java.nio.charset.StandardCharsets;
import java.security.KeyStore;

import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.GCMParameterSpec;

/** Stores personal text encrypted by an app-specific Android Keystore key. */
public final class UserProfileStore {
    private static final String KEY_ALIAS = "personacare-profile";
    private static final String ANDROID_KEYSTORE = "AndroidKeyStore";
    private static final String PREFERENCES = "user-profile";
    private final SharedPreferences preferences;

    public UserProfileStore(Context context) {
        preferences = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE);
    }

    public UserProfile load() {
        try {
            String encoded = preferences.getString("ciphertext", null);
            String encodedIv = preferences.getString("iv", null);
            if (encoded != null && encodedIv != null) {
                Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
                cipher.init(
                        Cipher.DECRYPT_MODE,
                        getOrCreateKey(),
                        new GCMParameterSpec(128, Base64.decode(encodedIv, Base64.NO_WRAP)));
                String json = new String(
                        cipher.doFinal(Base64.decode(encoded, Base64.NO_WRAP)),
                        StandardCharsets.UTF_8);
                JSONObject value = new JSONObject(json);
                return new UserProfile(
                        value.optString("person_name"),
                        value.optString("profile_text"),
                        value.optString("persona_id", UserProfile.PERSONA_BALANCED));
            }
            return migrateLegacyProfile();
        } catch (Exception invalidOrUnavailable) {
            return null;
        }
    }

    public void save(UserProfile profile) throws Exception {
        JSONObject value = new JSONObject()
                .put("person_name", profile.personName())
                .put("profile_text", profile.profileText())
                .put("persona_id", profile.personaId());
        Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
        cipher.init(Cipher.ENCRYPT_MODE, getOrCreateKey());
        byte[] ciphertext = cipher.doFinal(value.toString().getBytes(StandardCharsets.UTF_8));
        preferences.edit()
                .clear()
                .putString("ciphertext", Base64.encodeToString(ciphertext, Base64.NO_WRAP))
                .putString("iv", Base64.encodeToString(cipher.getIV(), Base64.NO_WRAP))
                .apply();
    }

    public void clear() {
        preferences.edit().clear().apply();
    }

    private UserProfile migrateLegacyProfile() throws Exception {
        String name = preferences.getString("person_name", "");
        if (name == null || name.trim().isEmpty()) {
            return null;
        }
        String style = preferences.getString("response_style", UserProfile.PERSONA_BALANCED);
        UserProfile profile = new UserProfile(name, "", style);
        save(profile);
        return profile;
    }

    private SecretKey getOrCreateKey() throws Exception {
        KeyStore keyStore = KeyStore.getInstance(ANDROID_KEYSTORE);
        keyStore.load(null);
        if (keyStore.containsAlias(KEY_ALIAS)) {
            return ((KeyStore.SecretKeyEntry) keyStore.getEntry(KEY_ALIAS, null)).getSecretKey();
        }
        KeyGenerator generator = KeyGenerator.getInstance(
                KeyProperties.KEY_ALGORITHM_AES, ANDROID_KEYSTORE);
        generator.init(new KeyGenParameterSpec.Builder(
                KEY_ALIAS,
                KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .build());
        return generator.generateKey();
    }
}
