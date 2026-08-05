package com.dragonnest.agent;

import com.dragonnest.proto.PersonalProfileRegistration;

/** User-facing profile values. Technical steering fields remain an implementation detail. */
public record UserProfile(String personName, String profileText, String personaId) {
    public static final String PERSONA_BALANCED = "balanced";
    public static final String PERSONA_CONCISE = "concise";
    public static final String PERSONA_DETAILED = "detailed";
    private static final String STYLE_VECTOR = "concise-vs-verbose-layer-7";

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
    }

    public PersonalProfileRegistration registration() {
        PersonalProfileRegistration.Builder registration = PersonalProfileRegistration.newBuilder()
                .setPersonName(personName)
                .setPreferredMode("auto")
                .setNotes(profileText)
                .setPersonaId(personaId)
                .setSteeringPositions("last")
                .setAllowRemoteVector(false);
        if (personaId.equals(PERSONA_CONCISE)) {
            registration.setSteeringVectorId(STYLE_VECTOR).setSteeringAlpha(-2.0f);
        } else if (personaId.equals(PERSONA_DETAILED)) {
            registration.setSteeringVectorId(STYLE_VECTOR).setSteeringAlpha(2.0f);
        }
        return registration.build();
    }
}
