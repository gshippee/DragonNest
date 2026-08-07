package com.dragonnest.agent;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public final class UserProfileTest {
    @Test
    public void sendsOnlySemanticPersonaRegistrationPolicy() {
        UserProfile concise = new UserProfile("Alex", "Prefers examples", "concise");
        UserProfile detailed = new UserProfile("Alex", "Prefers detail", "detailed");
        UserProfile balanced = new UserProfile("Alex", "", "balanced");

        assertEquals("concise", concise.registration().getPersonaId());
        assertEquals("", concise.registration().getSteeringVectorId());
        assertEquals(0.0f, concise.registration().getSteeringAlpha(), 0.0f);
        assertEquals("Prefers examples", concise.registration().getNotes());
        assertEquals("detailed", detailed.registration().getPersonaId());
        assertEquals("", detailed.registration().getSteeringVectorId());
        assertEquals(0.0f, detailed.registration().getSteeringAlpha(), 0.0f);
        assertEquals("", balanced.registration().getSteeringVectorId());
    }

    @Test
    public void validatesProfileLimitsAndPersona() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new UserProfile("", "", "balanced"));
        assertThrows(
                IllegalArgumentException.class,
                () -> new UserProfile("Alex", "x".repeat(501), "balanced"));
        assertThrows(
                IllegalArgumentException.class,
                () -> new UserProfile("Alex", "", "unknown"));
        assertTrue(new UserProfile(" Alex ", " Notes ", "balanced")
                .personName().equals("Alex"));
    }
}
