"""Compare TIFF conversion against the imported baseline in separate Windows processes."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import runpy
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def peak_memory_mib():
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
            "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize / 1024 ** 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "optimized"])
    parser.add_argument("--baseline-commit", default="dd0e9d3")
    args = parser.parse_args()
    directory = ROOT / ".build" / "benchmark"
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "input.tif"
    if args.mode:
        if args.mode == "baseline":
            convert = runpy.run_path(str(directory / "baseline.py"))["convert_tif_to_rgb"]
        else:
            from imageprocession.tif_rgb_converter import convert_tif_to_rgb as convert
        started = time.perf_counter()
        convert(str(source), str(directory / f"{args.mode}.tif"), "RGB TIF", stretch_percent=0)
        print(json.dumps({"mode": args.mode, "seconds": round(time.perf_counter() - started, 3),
                          "peak_working_set_mib": round(peak_memory_mib(), 2)}))
        return

    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    baseline = subprocess.run(["git", "show", f"{args.baseline_commit}:imageprocession/tif_rgb_converter.py"],
                              cwd=ROOT, check=True, stdout=subprocess.PIPE).stdout
    (directory / "baseline.py").write_bytes(baseline)
    with rasterio.open(source, "w", driver="GTiff", height=4096, width=4096, count=3, dtype="float32",
                       crs="EPSG:3857", transform=from_origin(100, 5000, 1, 1), tiled=True,
                       blockxsize=512, blockysize=512) as output:
        for y in range(0, 4096, 512):
            for x in range(0, 4096, 512):
                ys, xs = np.mgrid[y:y + 512, x:x + 512]
                block = np.stack([(xs + ys) % 4096, (xs * 2 + ys) % 8192, (xs + ys * 3) % 8192]).astype(np.float32)
                output.write(block, window=Window(x, y, 512, 512))
    results = []
    for mode in ("baseline", "optimized"):
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--mode", mode],
                                 cwd=ROOT, check=True, stdout=subprocess.PIPE, text=True)
        results.append(json.loads(process.stdout))
    result = {"fixture": "4096 x 4096, 3 bands, float32, exact min/max stretch, RGB GeoTIFF",
              "baseline_commit": args.baseline_commit, "measurements": results,
              "memory_reduction_percent": round((1 - results[1]["peak_working_set_mib"] / results[0]["peak_working_set_mib"]) * 100, 1)}
    (directory / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
