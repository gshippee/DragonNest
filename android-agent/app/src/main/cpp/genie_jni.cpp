#include <jni.h>

#include <Genie/GenieCommon.h>
#include <Genie/GenieDialog.h>

#include <sstream>
#include <string>

namespace {

struct QueryResult {
  std::string text;
};

void collectResponse(const char* response,
                     GenieDialog_SentenceCode_t,
                     const void* userData) {
  if (response == nullptr || userData == nullptr) {
    return;
  }
  auto* result = static_cast<QueryResult*>(const_cast<void*>(userData));
  result->text.append(response);
}

void throwIllegalState(JNIEnv* env, const std::string& message) {
  jclass exception = env->FindClass("java/lang/IllegalStateException");
  if (exception != nullptr) {
    env->ThrowNew(exception, message.c_str());
  }
}

bool createDialog(const std::string& config,
                  GenieDialogConfig_Handle_t* configHandle,
                  GenieDialog_Handle_t* dialogHandle,
                  std::string* error) {
  Genie_Status_t status = GenieDialogConfig_createFromJson(config.c_str(), configHandle);
  if (status != GENIE_STATUS_SUCCESS) {
    *error = "GenieDialogConfig_createFromJson failed with status " + std::to_string(status);
    return false;
  }
  status = GenieDialog_create(*configHandle, dialogHandle);
  if (status != GENIE_STATUS_SUCCESS) {
    GenieDialogConfig_free(*configHandle);
    *configHandle = nullptr;
    *error = "GenieDialog_create failed with status " + std::to_string(status);
    return false;
  }
  return true;
}

void freeDialog(GenieDialogConfig_Handle_t configHandle, GenieDialog_Handle_t dialogHandle) {
  if (dialogHandle != nullptr) {
    GenieDialog_free(dialogHandle);
  }
  if (configHandle != nullptr) {
    GenieDialogConfig_free(configHandle);
  }
}

std::string stringFromJni(JNIEnv* env, jstring value) {
  if (value == nullptr) {
    return "";
  }
  const char* raw = env->GetStringUTFChars(value, nullptr);
  if (raw == nullptr) {
    return "";
  }
  std::string result(raw);
  env->ReleaseStringUTFChars(value, raw);
  return result;
}

}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dragonnest_agent_vendor_GenieRuntimeBridge_nativeProbe(
    JNIEnv* env, jclass, jstring configJson) {
  GenieDialogConfig_Handle_t configHandle = nullptr;
  GenieDialog_Handle_t dialogHandle = nullptr;
  std::string error;
  const bool ready = createDialog(
      stringFromJni(env, configJson), &configHandle, &dialogHandle, &error);
  freeDialog(configHandle, dialogHandle);
  return ready ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_dragonnest_agent_vendor_GenieRuntimeBridge_nativeExecute(
    JNIEnv* env, jclass, jstring configJson, jstring prompt) {
  GenieDialogConfig_Handle_t configHandle = nullptr;
  GenieDialog_Handle_t dialogHandle = nullptr;
  std::string error;
  if (!createDialog(stringFromJni(env, configJson), &configHandle, &dialogHandle, &error)) {
    throwIllegalState(env, error);
    return nullptr;
  }

  QueryResult result;
  const std::string request = stringFromJni(env, prompt);
  const Genie_Status_t status = GenieDialog_query(
      dialogHandle,
      request.c_str(),
      GENIE_DIALOG_SENTENCE_COMPLETE,
      collectResponse,
      &result);
  freeDialog(configHandle, dialogHandle);
  if (status < GENIE_STATUS_SUCCESS) {
    throwIllegalState(env, "GenieDialog_query failed with status " + std::to_string(status));
    return nullptr;
  }
  return env->NewStringUTF(result.text.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_dragonnest_agent_vendor_GenieRuntimeBridge_nativeRuntimeVersion(
    JNIEnv* env, jclass) {
  std::ostringstream version;
  version << Genie_getApiMajorVersion() << '.' << Genie_getApiMinorVersion() << '.'
          << Genie_getApiPatchVersion();
  return env->NewStringUTF(version.str().c_str());
}
