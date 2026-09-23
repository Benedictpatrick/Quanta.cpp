// JNI bridge: dev.quanta.Native <-> Quanta engine (chat + on-device benchmark).
#include <android/log.h>
#include <jni.h>

#include <memory>
#include <string>
#include <vector>

#include "checks.h"
#include "model.h"
#include "session.h"
#include "tokenizer.h"

namespace {

const char* kTag = "Quanta";
const char* kSystem = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

struct Engine {
    quanta::Model model;
    std::unique_ptr<quanta::Tokenizer> tok;
    std::unique_ptr<quanta::Chat> chat;
    quanta::Chat::Stats last;
};

std::string to_string(JNIEnv* env, jstring s) {
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out(c);
    env->ReleaseStringUTFChars(s, c);
    return out;
}

// Java strings are UTF-16; GetStringUTFChars gives "modified UTF-8", which differs for emoji.
// Convert properly via String.getBytes("UTF-8").
std::string to_utf8(JNIEnv* env, jstring s) {
    jclass cls = env->GetObjectClass(s);
    jmethodID get_bytes = env->GetMethodID(cls, "getBytes", "(Ljava/lang/String;)[B");
    jstring enc = env->NewStringUTF("UTF-8");
    auto arr = static_cast<jbyteArray>(env->CallObjectMethod(s, get_bytes, enc));
    const jsize n = env->GetArrayLength(arr);
    std::string out(size_t(n), '\0');
    env->GetByteArrayRegion(arr, 0, n, reinterpret_cast<jbyte*>(out.data()));
    env->DeleteLocalRef(arr);
    env->DeleteLocalRef(enc);
    return out;
}

jbyteArray to_bytes(JNIEnv* env, const std::string& s) {
    jbyteArray arr = env->NewByteArray(jsize(s.size()));
    env->SetByteArrayRegion(arr, 0, jsize(s.size()), reinterpret_cast<const jbyte*>(s.data()));
    return arr;
}

// Length of the longest prefix of s that ends on a complete UTF-8 character (tokens can split characters).
size_t complete_utf8(const std::string& s) {
    size_t i = s.size(), back = 0;
    while (i > 0 && back < 4) {
        const unsigned char c = static_cast<unsigned char>(s[i - 1]);
        if ((c & 0xC0) != 0x80) {  // lead byte (or ASCII)
            const size_t need = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
            return back + 1 >= need ? s.size() : i - 1;
        }
        --i;
        ++back;
    }
    return s.size();
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_dev_quanta_Native_load(JNIEnv* env, jclass, jstring path, jint threads) {
    auto e = std::make_unique<Engine>();
    std::string err;
    if (!e->model.load(to_string(env, path), 4096, &err, threads)) {
        __android_log_print(ANDROID_LOG_ERROR, kTag, "load failed: %s", err.c_str());
        return 0;
    }
    e->tok = std::make_unique<quanta::Tokenizer>(e->model.file().tokenizer());
    quanta::SamplerParams sp;  // Qwen chat defaults
    e->chat = std::make_unique<quanta::Chat>(e->model, *e->tok, kSystem, sp);
    return reinterpret_cast<jlong>(e.release());
}

JNIEXPORT void JNICALL Java_dev_quanta_Native_free(JNIEnv*, jclass, jlong h) {
    delete reinterpret_cast<Engine*>(h);
}

JNIEXPORT void JNICALL Java_dev_quanta_Native_reset(JNIEnv*, jclass, jlong h) {
    reinterpret_cast<Engine*>(h)->chat->reset();
}

// callback: object with `boolean onPiece(byte[] utf8)`; return false to stop generating.
JNIEXPORT jboolean JNICALL Java_dev_quanta_Native_reply(JNIEnv* env, jclass, jlong h, jstring user, jint max_tokens,
                                                        jobject callback) {
    Engine* e = reinterpret_cast<Engine*>(h);
    jmethodID on_piece = env->GetMethodID(env->GetObjectClass(callback), "onPiece", "([B)Z");
    std::string pending;
    auto emit = [&](const std::string& bytes) {
        jbyteArray arr = to_bytes(env, bytes);
        const bool more = env->CallBooleanMethod(callback, on_piece, arr);
        env->DeleteLocalRef(arr);
        return more;
    };
    const bool ok = e->chat->reply(
        to_utf8(env, user), max_tokens,
        [&](const std::string& piece) {
            pending += piece;
            const size_t n = complete_utf8(pending);
            if (n == 0) return true;
            const bool more = emit(pending.substr(0, n));
            pending.erase(0, n);
            return more;
        },
        &e->last);
    if (!pending.empty()) emit(pending);
    return ok;
}

// Stats of the last reply: {prompt_tokens, prefill_seconds, reply_tokens, decode_seconds, ctx_used}
JNIEXPORT jdoubleArray JNICALL Java_dev_quanta_Native_lastStats(JNIEnv* env, jclass, jlong h) {
    const auto& s = reinterpret_cast<Engine*>(h)->last;
    const double v[5] = {double(s.prompt_tokens), s.prefill_seconds, double(s.reply_tokens), s.decode_seconds,
                         double(s.ctx_used)};
    jdoubleArray arr = env->NewDoubleArray(5);
    env->SetDoubleArrayRegion(arr, 0, 5, v);
    return arr;
}

// callback: object with `void onLine(String line)`.
JNIEXPORT jboolean JNICALL Java_dev_quanta_Native_benchmark(JNIEnv* env, jclass, jstring path, jdouble max_seconds,
                                                            jintArray threads, jobject callback) {
    jmethodID on_line = env->GetMethodID(env->GetObjectClass(callback), "onLine", "(Ljava/lang/String;)V");
    quanta::BenchOptions opt;
    opt.max_seconds = max_seconds;
    const jsize n = env->GetArrayLength(threads);
    opt.thread_counts.resize(size_t(n));
    env->GetIntArrayRegion(threads, 0, n, opt.thread_counts.data());
    const quanta::Log log = [&](const std::string& line) {
        __android_log_print(ANDROID_LOG_INFO, kTag, "%s", line.c_str());
        jbyteArray arr = to_bytes(env, line);
        jclass str_cls = env->FindClass("java/lang/String");
        jmethodID ctor = env->GetMethodID(str_cls, "<init>", "([BLjava/lang/String;)V");
        jstring enc = env->NewStringUTF("UTF-8");
        auto js = static_cast<jstring>(env->NewObject(str_cls, ctor, arr, enc));
        env->CallVoidMethod(callback, on_line, js);
        env->DeleteLocalRef(js);
        env->DeleteLocalRef(enc);
        env->DeleteLocalRef(str_cls);
        env->DeleteLocalRef(arr);
    };
    return quanta::run_benchmark(to_string(env, path), opt, log);
}

}  // extern "C"
