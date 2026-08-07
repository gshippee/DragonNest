package com.dragonnest.agent;

import com.dragonnest.proto.PersonalProfileRegistration;

/** User-facing profile values, including the steering strength the user chose. */
public record UserProfile(
        String personName, String profileText, String personaId, float steeringAlpha) {
    public static final String PERSONA_BALANCED = "balanced";
    public static final String PERSONA_CONCISE = "concise";
    public static final String PERSONA_DETAILED = "detailed";
    /**
     * Bounds of the profile steering slider. These match the validated alpha
     * range of concise-vs-verbose-layer-7; Brain rejects anything outside it,
     * so the UI must not offer a value that would fail at routing time.
     */
    public static final float ALPHA_MIN = -10.0f;
    public static final float ALPHA_MAX = 10.0f;

    /** Backwards-compatible constructor: no explicit strength means profile default. */
    public UserProfile(String personName, String profileText, String personaId) {
        this(personName, profileText, personaId, 0.0f);
    }

    public UserProfile {
        personName = personName == null ? "" : personName.trim();
        profileText = profileText == null ? "" : profileText.trim();
        personaId = personaId == null ? PERSONA_BALANCED : personaId.trim();
        if (personName.isEmpty() || personName.length() > 120) {
            throw new IllegalArgumentException("Enter your name");
        }
        if (profileText.length() > 500) {
            throw new IllegalArgumentException("About you must be 500 characters or fewer");
        }
        if (!personaId.equals(PERSONA_BALANCED)
                && !personaId.equals(PERSONA_CONCISE)
                && !personaId.equals(PERSONA_DETAILED)) {
            throw new IllegalArgumentException("Choose an available persona");
        }
        if (Float.isNaN(steeringAlpha) || steeringAlpha < ALPHA_MIN || steeringAlpha > ALPHA_MAX) {
            throw new IllegalArgumentException("Steering strength is out of range");
        }
    }

    /** True when the user moved the slider off centre and wants an explicit alpha. */
    public boolean hasExplicitSteering() {
        return Math.abs(steeringAlpha) > 0.01f;
    }

    public PersonalProfileRegistration registration() {
        return PersonalProfileRegistration.newBuilder()
                .setPersonName(personName)
                .setPreferredMode("auto")
                .setNotes(profileText)
                .setPersonaId(personaId)
                .build();
    }
}
