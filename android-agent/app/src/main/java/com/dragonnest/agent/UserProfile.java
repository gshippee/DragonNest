package com.dragonnest.agent;

import com.dragonnest.proto.PersonalProfileRegistration;

/** A small, client-owned profile. Technical steering fields are derived, not displayed. */
public record UserProfile(String personName, String preferredMode, String responseStyle) {
    public static final String STYLE_BALANCED = "balanced";
    public static final String STYLE_CONCISE = "concise";
    public static final String STYLE_DETAILED = "detailed";
    private static final String STYLE_VECTOR = "concise-vs-verbose-layer-7";

    public UserProfile {
        personName = personName == null ? "" : personName.trim();
        preferredMode = preferredMode == null ? "auto" : preferredMode.trim();
        responseStyle = responseStyle == null ? STYLE_BALANCED : responseStyle.trim();
        if (personName.isEmpty() || personName.length() > 120) {
            throw new IllegalArgumentException("Please enter your name");
        }
        if (!preferredMode.equals("auto")
                && !preferredMode.equals("fast")
                && !preferredMode.equals("private")
                && !preferredMode.equals("quality")) {
            throw new IllegalArgumentException("Choose how DragonNest should run");
        }
        if (!responseStyle.equals(STYLE_BALANCED)
                && !responseStyle.equals(STYLE_CONCISE)
                && !responseStyle.equals(STYLE_DETAILED)) {
            throw new IllegalArgumentException("Choose an answer style");
        }
    }

    public PersonalProfileRegistration registration() {
        PersonalProfileRegistration.Builder registration = PersonalProfileRegistration.newBuilder()
                .setPersonName(personName)
                .setPreferredMode(preferredMode)
                .setSteeringPositions("last")
                .setAllowRemoteVector(false);
        if (responseStyle.equals(STYLE_CONCISE)) {
            registration.setSteeringVectorId(STYLE_VECTOR).setSteeringAlpha(-2.0f);
        } else if (responseStyle.equals(STYLE_DETAILED)) {
            registration.setSteeringVectorId(STYLE_VECTOR).setSteeringAlpha(2.0f);
        }
        return registration.build();
    }
}
