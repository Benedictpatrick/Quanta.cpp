// JNI entry for dev.quanta.LlamaBench (one of libllamabench_v80/_v82/_v86.so is loaded, by CPU features).
#include <android/log.h>
#include <jni.h>

#include "llama_harness.h"

namespace {
std::string str(JNIEnv* env, jstring s) {
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out(c);
    env->ReleaseStringUTFChars(s, c);
    return out;
}
}  // namespace

extern "C" JNIEXPORT jboolean JNICALL Java_dev_quanta_LlamaBench_run(JNIEnv* env, jclass, jstring model, jstring label,
                                                                     jstring prompt, jintArray threads,
                                                                     jobject callback) {
    jmethodID on_line = env->GetMethodID(env->GetObjectClass(callback), "onLine", "(Ljava/lang/String;)V");
    std::vector<int> t(size_t(env->GetArrayLength(threads)));
    env->GetIntArrayRegion(threads, 0, jsize(t.size()), t.data());
    return llama_harness_run(str(env, model), str(env, label), str(env, prompt), t, [&](const std::string& line) {
        __android_log_print(ANDROID_LOG_INFO, "Quanta", "%s", line.c_str());
        jstring js = env->NewStringUTF(line.c_str());  // ASCII only
        env->CallVoidMethod(callback, on_line, js);
        env->DeleteLocalRef(js);
    });
}
