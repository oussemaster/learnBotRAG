"""
tests/test_tracker.py
======================
Unit-Tests für src/core/tracker.py.

Isolationsstrategie:
  - Alle Dateioperationen laufen ausschließlich in ``tmp_path`` (pytest-Fixture).
  - ``PROCESSED_FILES_PATH`` und ``VECTOR_DB_PATH`` werden per ``monkeypatch``
    auf tmp_path-basierte Pfade umgebogen – kein Zugriff auf das echte Projekt-
    Verzeichnis, keine Seiteneffekte zwischen Tests.
  - ``delete_vectors_fn`` wird als ``MagicMock`` übergeben, damit cleanup_deleted_files
    vollständig ohne ChromaDB getestet werden kann.
"""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from src.core.tracker import (
    build_registry_for_paths,
    cleanup_deleted_files,
    compute_deleted_files,
    compute_ingest_delta,
    file_content_hash,
    load_processed_registry,
    merge_into_registry,
    paths_for_tracker_keys,
    remove_keys_from_registry,
    save_processed_registry,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def patched_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Biegt VECTOR_DB_PATH und PROCESSED_FILES_PATH auf tmp_path um.

    Gibt das Tupel ``(vector_db_path, tracker_path)`` zurück, damit Tests
    die Pfade für Assertions verwenden können.
    """
    vector_db = tmp_path / "vector_db"
    tracker   = vector_db / "processed_files.json"

    monkeypatch.setattr("src.core.tracker.VECTOR_DB_PATH",       vector_db)
    monkeypatch.setattr("src.core.tracker.PROCESSED_FILES_PATH", tracker)

    return vector_db, tracker


@pytest.fixture()
def sample_file(tmp_path: Path) -> Path:
    """Schreibt eine einfache Textdatei und gibt ihren Pfad zurück."""
    p = tmp_path / "sample.csv"
    p.write_text("col1,col2\nfoo,bar\n", encoding="utf-8")
    return p


# ══════════════════════════════════════════════════════════════════════════════
# file_content_hash
# ══════════════════════════════════════════════════════════════════════════════

class TestFileContentHash:
    def test_known_value_empty_file(self, tmp_path: Path) -> None:
        """SHA-256 einer leeren Datei ist ein bekannter, fester Wert."""
        empty = tmp_path / "empty.bin"
        empty.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert file_content_hash(empty) == expected

    def test_deterministic(self, sample_file: Path) -> None:
        """Zwei Aufrufe auf derselben Datei liefern denselben Hash."""
        assert file_content_hash(sample_file) == file_content_hash(sample_file)

    def test_content_sensitive(self, tmp_path: Path) -> None:
        """Unterschiedlicher Inhalt → unterschiedlicher Hash."""
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"hello")
        f2.write_bytes(b"world")
        assert file_content_hash(f1) != file_content_hash(f2)

    def test_same_content_different_name(self, tmp_path: Path) -> None:
        """Gleicher Inhalt, anderer Dateiname → gleicher Hash (inhaltsbasiert)."""
        f1 = tmp_path / "x.csv"
        f2 = tmp_path / "y.pdf"
        content = b"identical content"
        f1.write_bytes(content)
        f2.write_bytes(content)
        assert file_content_hash(f1) == file_content_hash(f2)

    def test_binary_content(self, tmp_path: Path) -> None:
        """Funktioniert auch mit binären Dateien."""
        f = tmp_path / "data.bin"
        f.write_bytes(bytes(range(256)))
        result = file_content_hash(f)
        # Hexdigest muss exakt 64 Zeichen lang sein (SHA-256)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


# ══════════════════════════════════════════════════════════════════════════════
# load_processed_registry
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadProcessedRegistry:
    def test_returns_empty_dict_when_file_missing(
        self, patched_paths: tuple[Path, Path]
    ) -> None:
        """Fehlende Tracker-Datei → leeres Dict (kein Fehler)."""
        _, tracker = patched_paths
        assert not tracker.exists()
        assert load_processed_registry() == {}

    def test_loads_valid_registry(
        self, patched_paths: tuple[Path, Path]
    ) -> None:
        """Gültige JSON-Datei wird korrekt deserialisiert."""
        vector_db, tracker = patched_paths
        vector_db.mkdir(parents=True)
        expected = {"file_a.csv": "abc123", "notes.pdf": "def456"}
        tracker.write_text(json.dumps(expected), encoding="utf-8")

        result = load_processed_registry()
        assert result == expected

    def test_coerces_keys_and_values_to_str(
        self, patched_paths: tuple[Path, Path]
    ) -> None:
        """Zahlen als Keys/Values werden zu str normalisiert."""
        vector_db, tracker = patched_paths
        vector_db.mkdir(parents=True)
        # JSON mit numerischen Keys ist zwar invalid JSON-spec, aber Python
        # erlaubt es beim Dump mit int-Keys nicht – wir testen int-Values.
        tracker.write_text('{"file.csv": 42}', encoding="utf-8")

        result = load_processed_registry()
        assert result == {"file.csv": "42"}

    def test_returns_empty_dict_for_invalid_json(
        self, patched_paths: tuple[Path, Path]
    ) -> None:
        """Beschädigte JSON-Datei → leeres Dict (keine Exception)."""
        vector_db, tracker = patched_paths
        vector_db.mkdir(parents=True)
        tracker.write_text("{ this is not valid json }", encoding="utf-8")

        assert load_processed_registry() == {}

    def test_returns_empty_dict_for_non_dict_json(
        self, patched_paths: tuple[Path, Path]
    ) -> None:
        """Gültiges JSON, aber kein Objekt (z. B. Liste) → leeres Dict."""
        vector_db, tracker = patched_paths
        vector_db.mkdir(parents=True)
        tracker.write_text('["a", "b", "c"]', encoding="utf-8")

        assert load_processed_registry() == {}

    def test_returns_empty_dict_for_empty_object(
        self, patched_paths: tuple[Path, Path]
    ) -> None:
        """Leeres JSON-Objekt ``{}`` wird als leeres Dict zurückgegeben."""
        vector_db, tracker = patched_paths
        vector_db.mkdir(parents=True)
        tracker.write_text("{}", encoding="utf-8")

        assert load_processed_registry() == {}


# ══════════════════════════════════════════════════════════════════════════════
# save_processed_registry
# ══════════════════════════════════════════════════════════════════════════════

class TestSaveProcessedRegistry:
    def test_creates_directory_if_missing(
        self, patched_paths: tuple[Path, Path]
    ) -> None:
        """VECTOR_DB_PATH wird angelegt, wenn er noch nicht existiert."""
        vector_db, _ = patched_paths
        assert not vector_db.exists()

        save_processed_registry({"file.csv": "hash1"})
        assert vector_db.is_dir()

    def test_writes_valid_json(
        self, patched_paths: tuple[Path, Path]
    ) -> None:
        """Persistierte Datei ist gültiges, lesbares JSON."""
        _, tracker = patched_paths
        registry = {"b.pdf": "hash2", "a.csv": "hash1"}
        save_processed_registry(registry)

        loaded = json.loads(tracker.read_text(encoding="utf-8"))
        assert loaded == registry

    def test_output_is_sorted(
        self, patched_paths: tuple[Path, Path]
    ) -> None:
        """Keys werden alphabetisch sortiert gespeichert (sort_keys=True)."""
        _, tracker = patched_paths
        save_processed_registry({"z.csv": "h1", "a.csv": "h2", "m.pdf": "h3"})

        raw = tracker.read_text(encoding="utf-8")
        keys_in_file = [
            line.strip().split(":")[0].strip('" ')
            for line in raw.splitlines()
            if ":" in line
        ]
        assert keys_in_file == sorted(keys_in_file)

    def test_overwrites_existing_file(
        self, patched_paths: tuple[Path, Path]
    ) -> None:
        """Zweiter Aufruf überschreibt den vorherigen Stand vollständig."""
        _, tracker = patched_paths
        save_processed_registry({"old.csv": "old_hash"})
        save_processed_registry({"new.csv": "new_hash"})

        loaded = json.loads(tracker.read_text(encoding="utf-8"))
        assert loaded == {"new.csv": "new_hash"}
        assert "old.csv" not in loaded

    def test_roundtrip_with_load(
        self, patched_paths: tuple[Path, Path]
    ) -> None:
        """save → load ergibt dasselbe Dict (Roundtrip-Test)."""
        registry = {"alpha.csv": "aaa", "beta.pdf": "bbb"}
        save_processed_registry(registry)
        assert load_processed_registry() == registry


# ══════════════════════════════════════════════════════════════════════════════
# build_registry_for_paths
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildRegistryForPaths:
    def test_empty_list_returns_empty_dict(self) -> None:
        assert build_registry_for_paths([]) == {}

    def test_single_file_uses_filename_as_key(self, sample_file: Path) -> None:
        result = build_registry_for_paths([sample_file])
        assert "sample.csv" in result

    def test_value_matches_file_content_hash(self, sample_file: Path) -> None:
        result = build_registry_for_paths([sample_file])
        assert result["sample.csv"] == file_content_hash(sample_file)

    def test_multiple_files(self, tmp_path: Path) -> None:
        """Mehrere Dateien erzeugen mehrere Einträge mit korrekten Hashes."""
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.pdf"
        f1.write_bytes(b"content_a")
        f2.write_bytes(b"content_b")

        result = build_registry_for_paths([f1, f2])

        assert len(result) == 2
        assert result["a.csv"] == file_content_hash(f1)
        assert result["b.pdf"] == file_content_hash(f2)


# ══════════════════════════════════════════════════════════════════════════════
# merge_into_registry
# ══════════════════════════════════════════════════════════════════════════════

class TestMergeIntoRegistry:
    def test_adds_new_entry(self, sample_file: Path) -> None:
        existing = {"old.csv": "oldhash"}
        result = merge_into_registry(existing, [sample_file])
        assert "old.csv" in result
        assert "sample.csv" in result

    def test_overwrites_changed_file_hash(self, sample_file: Path) -> None:
        """Gleicher Dateiname, anderer Hash → Hash wird aktualisiert."""
        outdated = {"sample.csv": "outdated_hash_000"}
        result = merge_into_registry(outdated, [sample_file])
        assert result["sample.csv"] == file_content_hash(sample_file)
        assert result["sample.csv"] != "outdated_hash_000"

    def test_does_not_mutate_original_registry(self, sample_file: Path) -> None:
        """Original-Dict wird nicht verändert (immutable Semantic)."""
        original = {"old.csv": "oldhash"}
        snapshot = dict(original)
        merge_into_registry(original, [sample_file])
        assert original == snapshot

    def test_empty_paths_returns_copy(self) -> None:
        """Keine neuen Pfade → gibt eine Kopie des Original-Dicts zurück."""
        existing = {"x.csv": "hash_x"}
        result = merge_into_registry(existing, [])
        assert result == existing
        assert result is not existing  # muss eine Kopie sein


# ══════════════════════════════════════════════════════════════════════════════
# remove_keys_from_registry
# ══════════════════════════════════════════════════════════════════════════════

class TestRemoveKeysFromRegistry:
    def test_removes_existing_keys(self) -> None:
        registry = {"a.csv": "h1", "b.pdf": "h2", "c.csv": "h3"}
        result = remove_keys_from_registry(registry, ["a.csv", "b.pdf"])
        assert result == {"c.csv": "h3"}

    def test_ignores_unknown_keys(self) -> None:
        """Nicht-existente Keys lösen keinen Fehler aus."""
        registry = {"a.csv": "h1"}
        result = remove_keys_from_registry(registry, ["ghost.pdf"])
        assert result == {"a.csv": "h1"}

    def test_empty_keys_list_returns_copy(self) -> None:
        registry = {"a.csv": "h1"}
        result = remove_keys_from_registry(registry, [])
        assert result == registry
        assert result is not registry

    def test_does_not_mutate_original(self) -> None:
        registry = {"a.csv": "h1", "b.pdf": "h2"}
        snapshot = dict(registry)
        remove_keys_from_registry(registry, ["a.csv"])
        assert registry == snapshot

    def test_remove_all_keys_returns_empty_dict(self) -> None:
        registry = {"a.csv": "h1", "b.pdf": "h2"}
        result = remove_keys_from_registry(registry, ["a.csv", "b.pdf"])
        assert result == {}


# ══════════════════════════════════════════════════════════════════════════════
# compute_ingest_delta
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeIngestDelta:
    def test_empty_registry_all_paths_are_delta(self, tmp_path: Path) -> None:
        """Leere Registry → alle Dateien müssen indexiert werden."""
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.pdf"
        f1.write_bytes(b"data_a")
        f2.write_bytes(b"data_b")

        delta = compute_ingest_delta([f1, f2], registry={})
        assert set(delta) == {f1, f2}

    def test_up_to_date_registry_yields_empty_delta(self, sample_file: Path) -> None:
        """Korrekte Hashes im Tracker → kein Delta."""
        registry = build_registry_for_paths([sample_file])
        delta = compute_ingest_delta([sample_file], registry)
        assert delta == []

    def test_changed_file_content_in_delta(self, tmp_path: Path) -> None:
        """Datei geändert (neuer Inhalt) → im Delta, obwohl Name bekannt ist."""
        f = tmp_path / "data.csv"
        f.write_bytes(b"original content")
        registry = build_registry_for_paths([f])

        # Inhalt verändern
        f.write_bytes(b"modified content")

        delta = compute_ingest_delta([f], registry)
        assert f in delta

    def test_new_file_not_in_registry_in_delta(self, tmp_path: Path) -> None:
        """Datei, die noch nicht im Tracker steht, muss indexiert werden."""
        existing = tmp_path / "known.csv"
        new_file = tmp_path / "new.pdf"
        existing.write_bytes(b"known")
        new_file.write_bytes(b"new")

        # Nur known.csv ist im Tracker
        registry = build_registry_for_paths([existing])
        delta = compute_ingest_delta([existing, new_file], registry)

        assert new_file in delta
        assert existing not in delta

    def test_no_files_returns_empty_delta(self) -> None:
        """Keine Dateien auf Disk → kein Delta."""
        assert compute_ingest_delta([], registry={"old.csv": "abc"}) == []


# ══════════════════════════════════════════════════════════════════════════════
# compute_deleted_files
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeDeletedFiles:
    def test_no_deletions_returns_empty_list(self, tmp_path: Path) -> None:
        """Alle Tracker-Dateien existieren auf Disk → keine Deletions."""
        f = tmp_path / "present.csv"
        f.write_bytes(b"x")
        registry = {"present.csv": "somehash"}

        assert compute_deleted_files(registry, [f]) == []

    def test_detects_single_deleted_file(self, tmp_path: Path) -> None:
        """Eine Datei im Tracker, die auf Disk fehlt, wird erkannt."""
        f = tmp_path / "present.csv"
        f.write_bytes(b"x")
        registry = {"present.csv": "h1", "deleted.pdf": "h2"}

        result = compute_deleted_files(registry, [f])
        assert result == ["deleted.pdf"]

    def test_detects_multiple_deleted_files_sorted(self) -> None:
        """Mehrere verwaiste Einträge werden sortiert zurückgegeben."""
        registry = {"z.csv": "h1", "a.pdf": "h2", "m.csv": "h3"}
        result = compute_deleted_files(registry, all_paths=[])
        assert result == ["a.pdf", "m.csv", "z.csv"]

    def test_all_files_deleted(self) -> None:
        """Leere all_paths → alle Registry-Keys gelten als gelöscht."""
        registry = {"x.csv": "hx", "y.pdf": "hy"}
        result = compute_deleted_files(registry, all_paths=[])
        assert set(result) == {"x.csv", "y.pdf"}

    def test_empty_registry_returns_empty_list(self, tmp_path: Path) -> None:
        """Leerer Tracker → nie etwas zu löschen."""
        f = tmp_path / "file.csv"
        f.write_bytes(b"x")
        assert compute_deleted_files({}, [f]) == []


# ══════════════════════════════════════════════════════════════════════════════
# paths_for_tracker_keys
# ══════════════════════════════════════════════════════════════════════════════

class TestPathsForTrackerKeys:
    def test_constructs_paths_under_data_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Keys werden als Dateinamen unter DATA_DIR aufgelöst."""
        fake_data_dir = tmp_path / "data"
        monkeypatch.setattr("src.core.tracker.DATA_DIR", fake_data_dir)

        result = paths_for_tracker_keys(["lecture.pdf", "grades.csv"])

        assert len(result) == 2
        assert result[0] == (fake_data_dir / "lecture.pdf").resolve()
        assert result[1] == (fake_data_dir / "grades.csv").resolve()

    def test_empty_keys_returns_empty_list(self) -> None:
        assert paths_for_tracker_keys([]) == []


# ══════════════════════════════════════════════════════════════════════════════
# cleanup_deleted_files
# ══════════════════════════════════════════════════════════════════════════════

class TestCleanupDeletedFiles:
    def test_no_deletions_callback_not_called(
        self, tmp_path: Path, patched_paths: tuple[Path, Path]
    ) -> None:
        """Alle Tracker-Dateien vorhanden → Callback wird nicht aufgerufen."""
        f = tmp_path / "present.csv"
        f.write_bytes(b"x")
        registry = {"present.csv": "somehash"}
        mock_delete = MagicMock()

        result = cleanup_deleted_files(registry, [f], mock_delete)

        mock_delete.assert_not_called()
        assert result == registry

    def test_no_deletions_returns_original_registry(
        self, tmp_path: Path, patched_paths: tuple[Path, Path]
    ) -> None:
        """Rückgabewert ist dasselbe Dict-Objekt (keine unnötige Kopie)."""
        f = tmp_path / "present.csv"
        f.write_bytes(b"x")
        registry = {"present.csv": "somehash"}
        mock_delete = MagicMock()

        result = cleanup_deleted_files(registry, [f], mock_delete)
        assert result is registry

    def test_deleted_file_removed_from_result(
        self, tmp_path: Path, patched_paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verwaister Eintrag fehlt im zurückgegebenen Registry."""
        present = tmp_path / "present.csv"
        present.write_bytes(b"x")
        registry = {"present.csv": "h1", "deleted.pdf": "h2"}
        mock_delete = MagicMock()

        # DATA_DIR muss auf tmp_path zeigen, damit paths_for_tracker_keys
        # einen sinnvollen (wenn auch nicht existierenden) Pfad liefert
        monkeypatch.setattr("src.core.tracker.DATA_DIR", tmp_path)

        result = cleanup_deleted_files(registry, [present], mock_delete)

        assert "deleted.pdf" not in result
        assert "present.csv" in result

    def test_callback_called_with_reconstructed_paths(
        self, tmp_path: Path, patched_paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Callback erhält eine Liste mit den absolut aufgelösten Pfaden."""
        registry = {"gone.pdf": "h1"}
        mock_delete = MagicMock()
        monkeypatch.setattr("src.core.tracker.DATA_DIR", tmp_path)

        cleanup_deleted_files(registry, all_paths=[], delete_vectors_fn=mock_delete)

        mock_delete.assert_called_once()
        paths_arg: list[Path] = mock_delete.call_args[0][0]
        assert len(paths_arg) == 1
        assert paths_arg[0] == (tmp_path / "gone.pdf").resolve()

    def test_tracker_persisted_after_cleanup(
        self, tmp_path: Path, patched_paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nach dem Cleanup wird der aktualisierte Tracker auf Disk geschrieben."""
        _, tracker_path = patched_paths
        registry = {"removed.csv": "h1", "kept.pdf": "h2"}
        present = tmp_path / "kept.pdf"
        present.write_bytes(b"x")
        mock_delete = MagicMock()
        monkeypatch.setattr("src.core.tracker.DATA_DIR", tmp_path)

        cleanup_deleted_files(registry, [present], mock_delete)

        assert tracker_path.is_file()
        persisted = json.loads(tracker_path.read_text(encoding="utf-8"))
        assert "removed.csv" not in persisted
        assert "kept.pdf" in persisted

    def test_multiple_deletions_callback_called_once(
        self, patched_paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Auch bei mehreren verwaisten Dateien wird der Callback genau einmal aufgerufen."""
        registry = {"a.csv": "h1", "b.csv": "h2", "c.pdf": "h3"}
        mock_delete = MagicMock()
        monkeypatch.setattr("src.core.tracker.DATA_DIR", Path("/fake/data"))

        cleanup_deleted_files(registry, all_paths=[], delete_vectors_fn=mock_delete)

        # Callback wird genau EINMAL mit allen Pfaden aufgerufen
        assert mock_delete.call_count == 1
        paths_arg = mock_delete.call_args[0][0]
        assert len(paths_arg) == 3