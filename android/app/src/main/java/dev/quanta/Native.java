package dev.quanta;

/** Bridge to the C++ engine (libquanta_jni.so). */
final class Native {
    static {
        System.loadLibrary("quanta_jni");
    }

    interface PieceCallback {
        /** UTF-8 bytes of the next piece of the reply (always whole characters). Return false to stop. */
        boolean onPiece(byte[] utf8);
    }

    interface LineCallback {
        void onLine(String line);
    }

    /** Returns a handle, or 0 on failure. threads <= 0 picks a default. */
    static native long load(String modelPath, int threads);

    static native void free(long handle);

    /** Forgets the conversation. */
    static native void reset(long handle);

    /** Streams the assistant's reply to `user`. Returns false if the message doesn't fit in the context. */
    static native boolean reply(long handle, String user, int maxTokens, PieceCallback cb);

    /** {prompt_tokens, prefill_seconds, reply_tokens, decode_seconds, ctx_used} of the last reply. */
    static native double[] lastStats(long handle);

    /** Runs kernel + exactness checks and speed tests; returns true if every check passed. */
    static native boolean benchmark(String modelPath, double maxSeconds, int[] threadCounts, LineCallback cb);

    private Native() {}
}
