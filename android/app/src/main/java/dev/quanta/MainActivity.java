package dev.quanta;

import android.app.Activity;
import android.app.ActivityManager;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.ParcelFileDescriptor;
import android.util.Log;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.view.inputmethod.EditorInfo;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

/**
 * Chat screen, plus a benchmark mode. Firebase Test Lab starts the app with the Game Loop intent
 * (com.google.intent.action.TEST_LOOP); then it runs the benchmark, writes the report to the file Test Lab
 * gives us (and to logcat, tag "Quanta"), and closes itself.
 */
public class MainActivity extends Activity {
    private static final String TAG = "Quanta";
    private static final String TEST_LOOP = "com.google.intent.action.TEST_LOOP";

    private static final int BG = Color.rgb(12, 14, 20);
    private static final int CARD = Color.rgb(28, 32, 44);
    private static final int ACCENT = Color.rgb(110, 140, 255);
    private static final int TEXT = Color.rgb(230, 232, 240);
    private static final int MUTED = Color.rgb(140, 146, 165);

    private final Handler ui = new Handler(Looper.getMainLooper());
    private LinearLayout messages;
    private ScrollView scroll;
    private TextView status;
    private EditText input;
    private Button send;
    private long engine;
    private volatile boolean busy;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        buildUi();
        Intent intent = getIntent();
        if (TEST_LOOP.equals(intent.getAction())) {
            runBenchmark(intent.getData(), true);
        } else {
            setBusy(true);
            new Thread(this::loadModel).start();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (engine != 0) Native.free(engine);
    }

    // ------------------------------------------------------------------ UI

    private int dp(float v) {
        return (int) TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, getResources().getDisplayMetrics());
    }

    private GradientDrawable rounded(int color, float radius) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(color);
        d.setCornerRadius(dp(radius));
        return d;
    }

    private void buildUi() {
        getWindow().setStatusBarColor(BG);
        getWindow().setNavigationBarColor(BG);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(BG);
        root.setPadding(dp(12), dp(12), dp(12), dp(12));

        LinearLayout header = new LinearLayout(this);
        header.setGravity(Gravity.CENTER_VERTICAL);
        TextView title = new TextView(this);
        title.setText("Quanta");
        title.setTextColor(TEXT);
        title.setTextSize(22);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        header.addView(title, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        Button bench = smallButton("Benchmark");
        bench.setOnClickListener(v -> { if (!busy) runBenchmark(null, false); });
        header.addView(bench);
        Button clear = smallButton("New chat");
        clear.setOnClickListener(v -> {
            if (busy || engine == 0) return;
            Native.reset(engine);
            messages.removeAllViews();
        });
        header.addView(clear);
        root.addView(header);

        status = new TextView(this);
        status.setTextColor(MUTED);
        status.setTextSize(12);
        status.setPadding(0, dp(4), 0, dp(8));
        root.addView(status);

        scroll = new ScrollView(this);
        messages = new LinearLayout(this);
        messages.setOrientation(LinearLayout.VERTICAL);
        scroll.addView(messages);
        root.addView(scroll, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));

        LinearLayout row = new LinearLayout(this);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(0, dp(8), 0, 0);
        input = new EditText(this);
        input.setHint("Message");
        input.setHintTextColor(MUTED);
        input.setTextColor(TEXT);
        input.setBackground(rounded(CARD, 20));
        input.setPadding(dp(14), dp(10), dp(14), dp(10));
        input.setMaxLines(5);
        input.setImeOptions(EditorInfo.IME_ACTION_SEND);
        input.setOnEditorActionListener((v, id, e) -> { sendMessage(); return true; });
        row.addView(input, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        send = smallButton("Send");
        send.setOnClickListener(v -> sendMessage());
        row.addView(send);
        root.addView(row);
        setContentView(root);
    }

    private Button smallButton(String label) {
        Button b = new Button(this);
        b.setText(label);
        b.setAllCaps(false);
        b.setTextColor(Color.WHITE);
        b.setBackground(rounded(ACCENT, 18));
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, dp(40));
        lp.setMarginStart(dp(8));
        b.setLayoutParams(lp);
        b.setPadding(dp(14), 0, dp(14), 0);
        return b;
    }

    private TextView addBubble(String text, boolean user, boolean mono) {
        TextView tv = new TextView(this);
        tv.setText(text);
        tv.setTextColor(TEXT);
        tv.setTextSize(mono ? 11 : 15);
        tv.setTextIsSelectable(true);
        if (mono) tv.setTypeface(Typeface.MONOSPACE);
        tv.setBackground(rounded(user ? ACCENT : CARD, 16));
        tv.setPadding(dp(12), dp(8), dp(12), dp(8));
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        lp.gravity = user ? Gravity.END : Gravity.START;
        lp.topMargin = dp(6);
        if (user) lp.setMarginStart(dp(48)); else lp.setMarginEnd(mono ? 0 : dp(48));
        messages.addView(tv, lp);
        scrollDown();
        return tv;
    }

    private void scrollDown() {
        scroll.post(() -> scroll.fullScroll(View.FOCUS_DOWN));
    }

    private void setBusy(boolean b) {
        busy = b;
        ui.post(() -> {
            send.setEnabled(!b);
            send.setAlpha(b ? 0.5f : 1f);
        });
    }

    private void setStatus(String s) {
        ui.post(() -> status.setText(s));
    }

    // ------------------------------------------------------------------ model

    /** Copies the model out of the APK once (it is stored uncompressed), so the engine can mmap it. */
    private File modelFile() throws Exception {
        File out = new File(getFilesDir(), "model.qnt");
        long size;
        try (AssetFileDescriptor fd = getAssets().openFd("model.qnt")) {
            size = fd.getLength();
        }
        if (out.exists() && out.length() == size) return out;
        File tmp = new File(getFilesDir(), "model.qnt.part");
        try (InputStream in = getAssets().open("model.qnt"); OutputStream os = new FileOutputStream(tmp)) {
            byte[] buf = new byte[1 << 20];
            long done = 0;
            int n;
            while ((n = in.read(buf)) > 0) {
                os.write(buf, 0, n);
                done += n;
                if ((done & ((32L << 20) - 1)) < n)
                    setStatus(String.format(Locale.US, "Preparing model… %d%%", done * 100 / size));
            }
        }
        if (!tmp.renameTo(out)) throw new Exception("could not move model file");
        return out;
    }

    private void loadModel() {
        try {
            setStatus("Preparing model…");
            File f = modelFile();
            setStatus("Loading…");
            long t0 = System.nanoTime();
            engine = Native.load(f.getAbsolutePath(), 0);
            if (engine == 0) throw new Exception("engine failed to load the model");
            setStatus(String.format(Locale.US, "Ready · Qwen2.5-0.5B · loaded in %.2f s · runs fully offline",
                    (System.nanoTime() - t0) / 1e9));
            setBusy(false);
        } catch (Exception e) {
            Log.e(TAG, "load", e);
            setStatus("Error: " + e.getMessage());
        }
    }

    private void sendMessage() {
        String text = input.getText().toString().trim();
        if (text.isEmpty() || busy || engine == 0) return;
        input.setText("");
        addBubble(text, true, false);
        TextView reply = addBubble("…", false, false);
        setBusy(true);
        new Thread(() -> {
            StringBuilder sb = new StringBuilder();
            boolean ok = Native.reply(engine, text, 512, utf8 -> {
                sb.append(new String(utf8, StandardCharsets.UTF_8));
                String now = sb.toString();
                ui.post(() -> { reply.setText(now); scrollDown(); });
                return true;
            });
            double[] s = Native.lastStats(engine);
            ui.post(() -> {
                if (!ok) reply.setText("(message too long)");
                status.setText(String.format(Locale.US,
                        "prompt %d tok @ %.1f tok/s · reply %d tok @ %.1f tok/s · context %d",
                        (int) s[0], s[0] / Math.max(s[1], 1e-9), (int) s[2], s[2] / Math.max(s[3], 1e-9), (int) s[4]));
            });
            setBusy(false);
        }).start();
    }

    // ------------------------------------------------------------------ benchmark

    private static String readFirstLine(String path) {
        try (RandomAccessFile f = new RandomAccessFile(path, "r")) {
            String s = f.readLine();
            return s == null ? "?" : s.trim();
        } catch (Exception e) {
            return "?";
        }
    }

    private String deviceInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append("device: ").append(Build.MANUFACTURER).append(' ').append(Build.MODEL)
                .append(" (").append(Build.DEVICE).append("), Android ").append(Build.VERSION.RELEASE)
                .append(" / API ").append(Build.VERSION.SDK_INT).append('\n');
        if (Build.VERSION.SDK_INT >= 31)
            sb.append("soc: ").append(Build.SOC_MANUFACTURER).append(' ').append(Build.SOC_MODEL).append('\n');
        sb.append("hardware: ").append(Build.HARDWARE).append(", abis: ")
                .append(String.join(",", Build.SUPPORTED_ABIS)).append('\n');
        ActivityManager.MemoryInfo mi = new ActivityManager.MemoryInfo();
        ((ActivityManager) getSystemService(ACTIVITY_SERVICE)).getMemoryInfo(mi);
        sb.append(String.format(Locale.US, "ram: %.1f GB total, %.1f GB free\n", mi.totalMem / 1e9, mi.availMem / 1e9));
        int cores = Runtime.getRuntime().availableProcessors();
        sb.append("cores: ").append(cores).append(", max MHz:");
        for (int i = 0; i < cores; i++) {
            String khz = readFirstLine("/sys/devices/system/cpu/cpu" + i + "/cpufreq/cpuinfo_max_freq");
            try {
                sb.append(' ').append(Long.parseLong(khz) / 1000);
            } catch (NumberFormatException e) {
                sb.append(" ?");
            }
        }
        return sb.toString();
    }

    /** gameLoop = started by Test Lab: write the report to `out` and close when done. */
    private void runBenchmark(Uri out, boolean gameLoop) {
        setBusy(true);
        messages.removeAllViews();
        TextView report = addBubble("", false, true);
        setStatus("Running benchmark… (about 2–6 minutes)");
        new Thread(() -> {
            OutputStream os = null;
            ParcelFileDescriptor pfd = null;
            StringBuilder all = new StringBuilder();
            Native.LineCallback line = s -> {
                Log.i(TAG, s);
                all.append(s).append('\n');
                String now = all.toString();
                ui.post(() -> { report.setText(now); scrollDown(); });
            };
            try {
                if (out != null) {
                    try {  // if Test Lab's results file can't be opened, still run: every line also goes to logcat
                        pfd = getContentResolver().openFileDescriptor(out, "w");
                        os = new FileOutputStream(pfd.getFileDescriptor());
                    } catch (Exception e) {
                        line.onLine("(results file not writable, logcat only: " + e + ")");
                    }
                }
                final OutputStream fos = os;
                Native.LineCallback both = s -> {
                    line.onLine(s);
                    if (fos != null) {
                        try {
                            fos.write((s + "\n").getBytes(StandardCharsets.UTF_8));
                            fos.flush();  // keep partial results if Test Lab times out
                        } catch (Exception ignored) {
                        }
                    }
                };
                for (String s : deviceInfo().split("\n")) both.onLine(s);
                long t0 = System.nanoTime();
                File f = modelFile();
                both.onLine(String.format(Locale.US, "model file ready in %.1f s (%d MB)",
                        (System.nanoTime() - t0) / 1e9, f.length() >> 20));
                int cores = Runtime.getRuntime().availableProcessors();
                int[] threads = cores > 2 ? new int[] {0, 2, cores} : new int[] {0};
                boolean ok = Native.benchmark(f.getAbsolutePath(), 600, threads, both);
                both.onLine("RESULT: " + (ok ? "ALL PASS" : "FAILURES"));
                setStatus(ok ? "Benchmark done: all checks passed" : "Benchmark done: some checks FAILED");
            } catch (Throwable e) {
                Log.e(TAG, "benchmark", e);
                line.onLine("error: " + e);
            } finally {
                try {
                    if (os != null) os.close();
                    if (pfd != null) pfd.close();
                } catch (Exception ignored) {
                }
            }
            if (gameLoop) {
                ui.post(this::finish);
            } else {
                setBusy(false);
                if (engine == 0) new Thread(this::loadModel).start();
            }
        }).start();
    }
}
