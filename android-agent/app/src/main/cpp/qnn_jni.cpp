#include <jni.h>

#include <QnnInterface.h>
#include <QnnSystemInterface.h>

#include <fstream>
#include <string>

namespace {

std::string to_string(JNIEnv* env, jstring value) {
    if (value == nullptr) return {};
    const char* chars = env->GetStringUTFChars(value, nullptr);
    std::string output(chars == nullptr ? "" : chars);
    if (chars != nullptr) env->ReleaseStringUTFChars(value, chars);
    return output;
}

void throw_unsupported(JNIEnv* env, const char* detail) {
    jclass type = env->FindClass("java/lang/UnsupportedOperationException");
    env->ThrowNew(type, detail);
}

}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dragonnest_agent_vendor_QnnRuntimeBridge_nativeExecutionReady(
        JNIEnv*, jclass) {
    // Flip only after the QAIRT context/session, graph, tensor, and KV-cache
    // bindings below have passed the physical S25 acceptance test.
    return JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dragonnest_agent_vendor_QnnRuntimeBridge_nativeProbe(
        JNIEnv* env, jclass, jstring context_path) {
    const std::string path = to_string(env, context_path);
    std::ifstream context(path, std::ios::binary);
    if (!context.good()) return JNI_FALSE;

    const QnnInterface_t** providers = nullptr;
    uint32_t provider_count = 0;
    const Qnn_ErrorHandle_t status =
            QnnInterface_getProviders(&providers, &provider_count);
    return status == QNN_SUCCESS && providers != nullptr && provider_count > 0
            ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_dragonnest_agent_vendor_QnnRuntimeBridge_nativeCreateSession(
        JNIEnv* env, jclass, jstring, jstring, jstring, jint) {
    // This JNI boundary deliberately links the direct QNN API. The remaining
    // physical handoff must bind provider/backend/device/context creation to
    // the exact QAIRT 2.45 context-binary ABI and confirm it under the S25's
    // packaged 2.45 libraries. Returning a fake handle would make the Agent
    // advertise execution that cannot retain KV state.
    throw_unsupported(
            env,
            "QNN context deserialization/session binding requires physical QAIRT validation");
    return 0;
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_dragonnest_agent_vendor_QnnRuntimeBridge_nativeExecute(
        JNIEnv* env, jclass, jlong, jint, jstring, jint, jstring, jstring,
        jintArray, jbyteArray, jboolean) {
    throw_unsupported(
            env,
            "QNN graph execution/KV tensor binding requires physical QAIRT validation");
    return nullptr;
}

extern "C" JNIEXPORT void JNICALL
Java_com_dragonnest_agent_vendor_QnnRuntimeBridge_nativeReleaseSession(
        JNIEnv*, jclass, jlong) {
    // Safe for the zero/uncreated handle used until the physical adapter is bound.
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_dragonnest_agent_vendor_QnnRuntimeBridge_nativeRuntimeVersion(
        JNIEnv* env, jclass) {
    return env->NewStringUTF("QAIRT-QNN-direct-JNI-unvalidated");
}
