"""Collect distribution license files and generate Windows executable metadata."""
from __future__ import annotations

import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Toolbox.version import APP_VERSION, APP_TITLE


def main():
    build = ROOT / ".build"
    licenses = build / "licenses"
    if licenses.exists():
        shutil.rmtree(licenses)
    licenses.mkdir(parents=True)
    manifest = []
    for distribution in sorted(importlib.metadata.distributions(), key=lambda item: item.metadata["Name"].casefold()):
        name = distribution.metadata["Name"]
        copied = []
        for file in distribution.files or []:
            if not any(word in file.name.casefold() for word in ("license", "licence", "copying", "notice")):
                continue
            source = Path(distribution.locate_file(file))
            if not source.is_file() or source.suffix.lower() in (".py", ".pyc", ".js"):
                continue
            target = licenses / f"{name}-{distribution.version}" / str(file).replace("/", "__").replace("\\", "__")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied.append(str(target.relative_to(licenses)).replace("\\", "/"))
        manifest.append({"name": name, "version": distribution.version,
                         "license": distribution.metadata.get("License-Expression") or distribution.metadata.get("License", ""),
                         "homepage": distribution.metadata.get("Home-page", ""), "files": copied})
    (licenses / "dependencies.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    version = tuple(int(part) for part in APP_VERSION.split(".")) + (0,)
    (build / "version_info.txt").write_text(f'''VSVersionInfo(
  ffi=FixedFileInfo(filevers={version!r}, prodvers={version!r}, mask=0x3f, flags=0x0,
                   OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[StringFileInfo([StringTable('080404B0', [
    StringStruct('CompanyName', 'yaluncoco'),
    StringStruct('FileDescription', {APP_TITLE!r}),
    StringStruct('FileVersion', {APP_VERSION!r}),
    StringStruct('InternalName', 'RemoteSensingToolbox'),
    StringStruct('OriginalFilename', 'RemoteSensingToolbox.exe'),
    StringStruct('ProductName', {APP_TITLE!r}),
    StringStruct('ProductVersion', {APP_VERSION!r})
  ])]), VarFileInfo([VarStruct('Translation', [2052, 1200])])]
)
''', encoding="utf-8")
    print(f"Prepared version {APP_VERSION} and license index for {len(manifest)} distributions.")


if __name__ == "__main__":
    main()
