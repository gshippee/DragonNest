package com.dragonnest.agent;

/** Resolves optional vendor bridge classes included by a signed Android runtime build. */
final class RuntimeBridgeLoader {
    private RuntimeBridgeLoader() { }

    static AndroidRuntimeBridge load(String runtime) {
        String[] classNames = switch (runtime) {
            case "qnn" -> new String[] {"com.dragonnest.agent.vendor.QnnRuntimeBridge"};
            case "genie" -> new String[] {
                    "com.dragonnest.agent.vendor.GenieXRuntimeBridge",
                    "com.dragonnest.agent.vendor.GenieRuntimeBridge"
            };
            // Runtime activation steering runs on a forked GenieX closure whose
            // sonames were renamed so it cannot displace the stock runtime
            // above. Keeping it behind its own runtime name is what makes the
            // physically accepted Base path immune to steering changes.
            case "genie_aux" -> new String[] {
                    "com.dragonnest.agent.vendor.GenieXSteeringRuntimeBridge"
            };
            default -> new String[0];
        };
        for (String className : classNames) {
            try {
                Class<?> candidate = Class.forName(className);
                Object instance = candidate.getDeclaredConstructor().newInstance();
                if (instance instanceof AndroidRuntimeBridge bridge
                        && runtime.equals(bridge.runtimeName())) {
                    return bridge;
                }
            } catch (ReflectiveOperationException | LinkageError unavailable) {
                // Optional runtime closures are intentionally absent from thin builds.
            }
        }
        return null;
    }
}
