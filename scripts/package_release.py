"""Create a portable archive and SHA-256 manifest for completed Windows artifacts."""
import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Toolbox.version import APP_VERSION


def main():
    release = ROOT / "release"
    release.mkdir(exist_ok=True)
    app = ROOT / "dist" / "RemoteSensingToolbox"
    if not (app / "RemoteSensingToolbox.exe").exists():
        raise FileNotFoundError("Build the Windows executable first.")
    archive = release / f"RemoteSensingToolbox-{APP_VERSION}-windows-x64-portable.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(app.rglob("*")):
            if path.is_file():
                output.write(path, path.relative_to(app.parent))
    lines = []
    for path in sorted(release.glob(f"RemoteSensingToolbox-{APP_VERSION}-windows-x64-*")):
        if path.suffix not in (".exe", ".zip"):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        lines.append(f"{digest.hexdigest()}  {path.name}")
    (release / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
