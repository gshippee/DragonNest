package com.dragonnest.agent;

/** Resolves optional vendor bridge classes included by a signed Android runtime build. */
final class RuntimeBridgeLoader {
    private RuntimeBridgeLoader() { }

    static AndroidRuntimeBridge load(String runtime) {
        String className = switch (runtime) {
            case "qnn" -> "com.dragonnest.agent.vendor.QnnRuntimeBridge";
            case "genie" -> "com.dragonnest.agent.vendor.GenieRuntimeBridge";
            default -> "";
        };
        if (className.isEmpty()) {
            return null;
        }
        try {
            Class<?> candidate = Class.forName(className);
            Object instance = candidate.getDeclaredConstructor().newInstance();
            if (!(instance instanceof AndroidRuntimeBridge bridge)) {
                return null;
            }
            return runtime.equals(bridge.runtimeName()) ? bridge : null;
        } catch (ReflectiveOperationException | LinkageError unavailable) {
            return null;
        }
    }
}
