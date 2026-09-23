#!/bin/sh
# Builds the llama.cpp comparison harness for Android arm64 in three CPU variants and copies them into the app:
#   v80: ARMv8.0 (Cortex-A53 phones, no dot product)
#   v82: ARMv8.2 + dotprod + fp16 (Cortex-A55/A76 and newer)
#   v86: ARMv8.6 + dotprod + fp16 + i8mm (e.g. Cortex-X4/A720 flagships)
# The app loads the best one the phone supports.
set -e
cd "$(dirname "$0")/.."
NDK=${ANDROID_NDK:-C:/Users/bened/AndroidSdk/ndk/28.2.13676358}
OUT=android/app/src/main/jniLibs/arm64-v8a
mkdir -p "$OUT"
for v in "v80 armv8-a" "v82 armv8.2-a+dotprod+fp16" "v86 armv8.6-a+dotprod+fp16+i8mm"; do
  set -- $v
  B=build-llamabench-android-$1
  cmake -S bench/llamabench -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_static \
    -DLLAMABENCH_VARIANT="$1" -DGGML_CPU_ARM_ARCH="$2" >/dev/null
  cmake --build "$B" --target "llamabench_$1"
  cp "$B/libllamabench_$1.so" "$OUT/"
done
ls -la "$OUT"
