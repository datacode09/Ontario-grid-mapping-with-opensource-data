"""Disk-based file cache with SHA-256 integrity checking and age expiry."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


class FileCache:
    """Simple on-disk cache for downloaded files.

    Parameters
    ----------
    cache_dir:
        Root directory for cached files.
    max_age_days:
        Maximum age before a cached entry is considered stale.
    """

    _MANIFEST = ".cache_manifest.json"

    def __init__(self, cache_dir: str | Path, max_age_days: int = 7) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_days = max_age_days
        self._manifest_path = self.cache_dir / self._MANIFEST
        self._manifest = self._load_manifest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Path]:
        """Return cached file path if present and not stale, else None."""
        entry = self._manifest.get(key)
        if not entry:
            return None

        path = Path(entry["path"])
        if not path.exists():
            del self._manifest[key]
            self._save_manifest()
            return None

        age_days = (time.time() - entry["timestamp"]) / 86400
        if age_days > self.max_age_days:
            return None  # Stale — caller should re-download

        return path

    def put(
        self,
        key: str,
        source_path: Path,
        url: Optional[str] = None,
        compute_hash: bool = True,
    ) -> Path:
        """Move/copy source_path into cache and record metadata."""
        safe_name = self._safe_filename(key, source_path)
        dest = self.cache_dir / safe_name
        dest.parent.mkdir(parents=True, exist_ok=True)

        if source_path != dest:
            shutil.copy2(str(source_path), str(dest))

        file_hash = self._sha256(dest) if compute_hash else ""

        self._manifest[key] = {
            "path": str(dest),
            "timestamp": time.time(),
            "downloaded_at": datetime.utcnow().isoformat() + "Z",
            "url": url or "",
            "sha256": file_hash,
        }
        self._save_manifest()
        return dest

    def is_fresh(self, key: str) -> bool:
        return self.get(key) is not None

    def invalidate(self, key: str) -> None:
        self._manifest.pop(key, None)
        self._save_manifest()

    def log_entry(self, key: str) -> Optional[dict]:
        return self._manifest.get(key)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict:
        if self._manifest_path.exists():
            try:
                return json.loads(self._manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_manifest(self) -> None:
        self._manifest_path.write_text(
            json.dumps(self._manifest, indent=2, ensure_ascii=False)
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _safe_filename(key: str, source: Path) -> str:
        safe = key.replace("://", "_").replace("/", "_").replace("?", "_")
        if len(safe) > 80:
            safe = hashlib.md5(safe.encode()).hexdigest()
        return safe + source.suffix
