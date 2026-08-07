cmake_minimum_required(VERSION 3.22.1)
project(dragonnest_vendor_jni LANGUAGES CXX)

if(NOT DEFINED QAIRT_SDK_ROOT OR QAIRT_SDK_ROOT STREQUAL "")
    message(FATAL_ERROR "QAIRT_SDK_ROOT is required for the DragonNest Genie JNI bridge")
endif()

set(QAIRT_INCLUDE_DIR "${QAIRT_SDK_ROOT}/include")
set(QAIRT_ANDROID_LIB_DIR "${QAIRT_SDK_ROOT}/lib/aarch64-android")
if(NOT EXISTS "${QAIRT_INCLUDE_DIR}/Genie/GenieDialog.h")
    message(FATAL_ERROR "QAIRT_SDK_ROOT does not contain Genie headers")
endif()
if(NOT EXISTS "${QAIRT_ANDROID_LIB_DIR}/libGenie.so")
    message(FATAL_ERROR "QAIRT_SDK_ROOT does not contain Android libGenie.so")
endif()
if(NOT EXISTS "${QAIRT_INCLUDE_DIR}/QNN/QnnInterface.h")
    message(FATAL_ERROR "QAIRT_SDK_ROOT does not contain QNN headers")
endif()
if(NOT EXISTS "${QAIRT_ANDROID_LIB_DIR}/libQnnHtp.so")
    message(FATAL_ERROR "QAIRT_SDK_ROOT does not contain Android libQnnHtp.so")
endif()
if(NOT EXISTS "${QAIRT_ANDROID_LIB_DIR}/libQnnSystem.so")
    message(FATAL_ERROR "QAIRT_SDK_ROOT does not contain Android libQnnSystem.so")
endif()

add_library(dragonnest_genie_jni SHARED genie_jni.cpp)
target_compile_features(dragonnest_genie_jni PRIVATE cxx_std_17)
target_include_directories(dragonnest_genie_jni PRIVATE "${QAIRT_INCLUDE_DIR}")

add_library(genie SHARED IMPORTED)
set_target_properties(genie PROPERTIES
    IMPORTED_LOCATION "${QAIRT_ANDROID_LIB_DIR}/libGenie.so")

find_library(android-log-lib log)
target_link_libraries(dragonnest_genie_jni PRIVATE genie ${android-log-lib})

add_library(dragonnest_qnn_jni SHARED qnn_jni.cpp)
target_compile_features(dragonnest_qnn_jni PRIVATE cxx_std_17)
target_include_directories(dragonnest_qnn_jni PRIVATE
    "${QAIRT_INCLUDE_DIR}"
    "${QAIRT_INCLUDE_DIR}/QNN")

add_library(qnn-htp SHARED IMPORTED)
set_target_properties(qnn-htp PROPERTIES
    IMPORTED_LOCATION "${QAIRT_ANDROID_LIB_DIR}/libQnnHtp.so")
add_library(qnn-system SHARED IMPORTED)
set_target_properties(qnn-system PROPERTIES
    IMPORTED_LOCATION "${QAIRT_ANDROID_LIB_DIR}/libQnnSystem.so")

find_library(android-dl-lib dl)
target_link_libraries(dragonnest_qnn_jni PRIVATE
    qnn-htp qnn-system ${android-log-lib} ${android-dl-lib})
