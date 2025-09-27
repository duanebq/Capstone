import os
import sys
import shutil
import zipfile
import tempfile
import urllib.request
import subprocess
from pathlib import Path

FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def pick_install_dir() -> Path:
    # Try C:\ffmpeg first, fall back to user's home if permission denied
    try_dir = Path("C:/ffmpeg")
    try:
        ensure_dir(try_dir / "bin")
        # quick write test
        testfile = try_dir / "bin" / ".write_test"
        testfile.write_text("ok", encoding="utf-8")
        testfile.unlink(missing_ok=True)
        return try_dir
    except Exception:
        home_dir = Path.home() / "ffmpeg"
        ensure_dir(home_dir / "bin")
        return home_dir

def download_ffmpeg_zip(tmp_dir: Path) -> Path:
    zip_path = tmp_dir / "ffmpeg-release-essentials.zip"
    print(f"⬇️  Downloading FFmpeg zip to: {zip_path}")
    urllib.request.urlretrieve(FFMPEG_URL, zip_path)
    return zip_path

def extract_zip(zip_path: Path, tmp_dir: Path) -> Path:
    print("📦 Extracting zip...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(tmp_dir)
    # Find the extracted root directory (contains /bin/ffmpeg.exe)
    for root, dirs, files in os.walk(tmp_dir):
        if root.lower().endswith("bin") and any(f.lower() == "ffmpeg.exe" for f in files):
            return Path(root).parent  # return the folder that contains 'bin'
    raise RuntimeError("Could not locate FFmpeg 'bin' folder inside the zip.")

def copy_bin(src_root: Path, dst_root: Path):
    src_bin = src_root / "bin"
    dst_bin = dst_root / "bin"
    ensure_dir(dst_bin)
    copied = []
    for exe in ("ffmpeg.exe", "ffprobe.exe", "ffplay.exe"):
        src = src_bin / exe
        if src.exists():
            shutil.copy2(src, dst_bin / exe)
            copied.append(exe)
    if not copied:
        raise RuntimeError("No FFmpeg executables found to copy.")
    print(f"✅ Copied: {', '.join(copied)} → {dst_bin}")

def add_to_path_for_process(dst_root: Path):
    bin_path = str(dst_root / "bin")
    os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")
    print(f"🔧 Added to current PATH: {bin_path}")

def verify_ffmpeg(dst_root: Path):
    ffmpeg_exe = str(dst_root / "bin" / "ffmpeg.exe")
    try:
        out = subprocess.run([ffmpeg_exe, "-version"], capture_output=True, text=True)
        if out.returncode == 0 and "ffmpeg version" in out.stdout.lower():
            print("🎉 FFmpeg is installed and responding:\n")
            print(out.stdout.splitlines()[0])
            return True
        else:
            print("⚠️ FFmpeg did not respond as expected.")
            print("stdout:\n", out.stdout)
            print("stderr:\n", out.stderr)
            return False
    except FileNotFoundError:
        print("❌ Could not run ffmpeg.exe (not found).")
        return False

def main():
    if os.name != "nt":
        print("This helper is tailored for Windows. On macOS/Linux, install ffmpeg via Homebrew/apt/etc.")
        return

    dst_root = pick_install_dir()
    print(f"🗂️  Target install dir: {dst_root}")

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        zip_path = download_ffmpeg_zip(tmp_dir)
        src_root = extract_zip(zip_path, tmp_dir)
        copy_bin(src_root, dst_root)

    add_to_path_for_process(dst_root)
    ok = verify_ffmpeg(dst_root)

    if ok:
        print("\n✅ Ready to use with yt-dlp.\n")
        print("If you want to add FFmpeg permanently to PATH (so all shells see it),")
        print("add this folder to your System/User PATH environment variable:")
        print(f"    {dst_root}\\bin\n")
        print("PowerShell (current session only):")
        print(f'    $env:Path = "{dst_root}\\bin;" + $env:Path')
    else:
        print("\n❌ FFmpeg verification failed. If you installed to a non-standard location,")
        print("   pass it to yt-dlp with:  --ffmpeg-location \"<path-to>\\bin\"")

if __name__ == "__main__":
    main()