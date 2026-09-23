# Testing Quanta on real phones with Firebase Test Lab (no phone needed)

The app `dist/quanta-0.1.apk` (458 MB, model included) has a benchmark mode that Test Lab starts
automatically ("Game Loop" test). It prints a full report: CPU info, kernel checks, prompt speed,
generation speed (full head vs shortlist head), exactness checks, and a thread-count sweep.

Free plan (Spark): 5 real-phone runs + 10 virtual-device runs per day.

## Steps
1. Open https://console.firebase.google.com and sign in with a Google account.
2. **Create a project** (any name, e.g. `quanta-bench`; Google Analytics can be off). Free plan is fine.
3. In the left menu open **Test Lab** (under *Run* or *Release & Monitor*) → **Run a test** → **Run a Game Loop test**.
4. Upload `E:\Phonecoode\dist\quanta-0.1.apk`.
5. **First run: virtual device** (free smoke test) — pick one virtual device (e.g. a Pixel, Android 12+).
   If you see a **timeout** option, set it to **15 minutes**. Start. It checks the app installs and runs.
6. **Then real phones** — pick *physical* devices, ideally one of each:
   - budget (e.g. a Galaxy A-series / Redmi, older chip),
   - mid-range (e.g. Pixel 6a / 7a),
   - flagship (e.g. Pixel 8/9, Galaxy S23/S24).
   Set timeout 15 min again if offered.
7. When finished, click each device → **Logs** tab (logcat). Download it (or copy the lines containing
   `Quanta`) and save it to `Downloads`. The video tab also shows the report on screen.
   Tell Claude the file names and it will read and analyse them.

## What the report contains (in order — most important first)
`device/soc/ram/cores` → `cpu features` (sdot/i8mm) → kernel checks (bit-exact PASS/FAIL + GB/s) → load →
prefill tok/s (batched vs token-by-token) → decode tok/s (full vs shortlist head, A-B-B-A order) →
exactness checks → decode with 2 threads / all cores → `RESULT: ALL PASS` or `FAILURES`.
If the run is cut off by a timeout, everything printed up to that point is still in the log.

## If something goes wrong
- Crash right after start on a budget phone (SIGILL in the log): the non-sdot kernel path has a bug.
- `load failed`: model copy or file problem (look at the lines before it).
- No `Quanta` lines at all: the app did not receive the Game Loop intent; check that the test type was Game Loop.
