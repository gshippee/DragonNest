cmake_minimum_required(VERSION 3.22.1)
project(dragonnest_genie_jni LANGUAGES CXX)

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

add_library(dragonnest_genie_jni SHARED genie_jni.cpp)
target_compile_features(dragonnest_genie_jni PRIVATE cxx_std_17)
target_include_directories(dragonnest_genie_jni PRIVATE "${QAIRT_INCLUDE_DIR}")

add_library(genie SHARED IMPORTED)
set_target_properties(genie PROPERTIES
    IMPORTED_LOCATION "${QAIRT_ANDROID_LIB_DIR}/libGenie.so")

find_library(android-log-lib log)
target_link_libraries(dragonnest_genie_jni PRIVATE genie ${android-log-lib})
