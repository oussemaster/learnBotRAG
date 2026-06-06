"""
tests/test_loaders.py
======================
Unit-Tests für src/core/loaders.py.

Isolationsstrategie:
  - ``discover_data_paths`` akzeptiert ``data_dir`` als Parameter → kein Patchen
    nötig, ``tmp_path`` wird direkt übergeben.
  - ``LOADER_BY_SUFFIX`` wird per ``patch.dict`` ersetzt.
    Warum patch.dict statt patch("src.core.loaders.CSVLoader"):
    ``LOADER_BY_SUFFIX`` ist ein Dict, das die echten Klassen bereits zum
    Import-Zeitpunkt als Werte einfriert. Ein Patch auf den Modulnamen
    ``CSVLoader`` ändert nur den Namen im Modul-Namespace, nicht die Referenz
    im Dict. ``patch.dict`` tauscht dagegen direkt den Dict-Inhalt aus und
    stellt ihn nach dem Test zuverlässig wieder her.
  - Jeder Test erzeugt nur die Dateien, die er wirklich braucht.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

import src.core.loaders as loaders_module
from src.core.loaders import LOADER_BY_SUFFIX, discover_data_paths, load_documents


# ── Hilfsfunktionen ────────────────────────────────────────────────────────────

def _make_doc(content: str, source: str) -> Document:
    """Erzeugt ein minimales Document-Objekt für Mock-Rückgaben."""
    return Document(page_content=content, metadata={"source": source})


# ══════════════════════════════════════════════════════════════════════════════
# LOADER_BY_SUFFIX – Factory-Konfiguration
# ══════════════════════════════════════════════════════════════════════════════

class TestLoaderBySuffix:
    def test_csv_suffix_registered(self) -> None:
        assert ".csv" in LOADER_BY_SUFFIX

    def test_pdf_suffix_registered(self) -> None:
        assert ".pdf" in LOADER_BY_SUFFIX

    def test_txt_not_registered(self) -> None:
        """Unbekannte Endungen sind bewusst nicht in der Factory."""
        assert ".txt" not in LOADER_BY_SUFFIX

    def test_all_values_are_classes(self) -> None:
        """Alle Einträge müssen aufrufbare Klassen sein (kein None, kein String)."""
        for suffix, cls in LOADER_BY_SUFFIX.items():
            assert callable(cls), f"Loader für '{suffix}' ist nicht aufrufbar"


# ══════════════════════════════════════════════════════════════════════════════
# discover_data_paths
# ══════════════════════════════════════════════════════════════════════════════

class TestDiscoverDataPaths:
    def test_nonexistent_directory_returns_empty_list(self, tmp_path: Path) -> None:
        """Nicht-existentes Verzeichnis → leere Liste, kein Crash."""
        missing = tmp_path / "does_not_exist"
        assert discover_data_paths(missing) == []

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        """Leeres Verzeichnis → leere Liste."""
        assert discover_data_paths(tmp_path) == []

    def test_finds_csv_files(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2")
        result = discover_data_paths(tmp_path)
        assert csv_file in result

    def test_finds_pdf_files(self, tmp_path: Path) -> None:
        pdf_file = tmp_path / "lecture.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")
        result = discover_data_paths(tmp_path)
        assert pdf_file in result

    def test_ignores_unsupported_extensions(self, tmp_path: Path) -> None:
        """Dateien mit unbekannten Endungen werden nicht zurückgegeben."""
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "notes.md").write_text("# Notes")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "archive.zip").write_bytes(b"PK\x03\x04")

        result = discover_data_paths(tmp_path)
        assert result == []

    def test_mixed_files_only_supported_returned(self, tmp_path: Path) -> None:
        """Nur .csv und .pdf werden zurückgegeben, der Rest nicht."""
        csv   = tmp_path / "table.csv"
        pdf   = tmp_path / "slides.pdf"
        txt   = tmp_path / "notes.txt"
        docx  = tmp_path / "report.docx"
        csv.write_bytes(b"x")
        pdf.write_bytes(b"x")
        txt.write_bytes(b"x")
        docx.write_bytes(b"x")

        result = discover_data_paths(tmp_path)
        assert set(result) == {csv, pdf}
        assert txt not in result
        assert docx not in result

    def test_result_is_sorted(self, tmp_path: Path) -> None:
        """Rückgabe ist stabil alphabetisch sortiert."""
        (tmp_path / "z_last.csv").write_bytes(b"x")
        (tmp_path / "a_first.pdf").write_bytes(b"x")
        (tmp_path / "m_middle.csv").write_bytes(b"x")

        result = discover_data_paths(tmp_path)
        assert result == sorted(result)

    def test_recursive_discovery(self, tmp_path: Path) -> None:
        """Dateien in Unterverzeichnissen werden gefunden (rglob)."""
        sub = tmp_path / "subfolder" / "deep"
        sub.mkdir(parents=True)
        nested_pdf = sub / "nested.pdf"
        nested_pdf.write_bytes(b"pdf content")

        result = discover_data_paths(tmp_path)
        assert nested_pdf in result

    def test_case_insensitive_suffix_matching(self, tmp_path: Path) -> None:
        """Groß- und Kleinschreibung der Endung ist egal (.CSV, .PDF)."""
        upper_csv = tmp_path / "DATA.CSV"
        upper_pdf = tmp_path / "SLIDES.PDF"
        upper_csv.write_bytes(b"x")
        upper_pdf.write_bytes(b"x")

        result = discover_data_paths(tmp_path)
        assert upper_csv in result
        assert upper_pdf in result

    def test_ignores_directories(self, tmp_path: Path) -> None:
        """Verzeichnisse mit passender Endung werden nicht als Dateien zurückgegeben."""
        fake_dir = tmp_path / "not_a_file.csv"
        fake_dir.mkdir()

        result = discover_data_paths(tmp_path)
        assert result == []


# ══════════════════════════════════════════════════════════════════════════════
# load_documents
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadDocuments:
    """
    Alle Tests patchen ``LOADER_BY_SUFFIX`` per ``patch.dict``.

    Hintergrund: ``LOADER_BY_SUFFIX`` ist ein Modul-Level-Dict, das die echten
    LangChain-Loader-Klassen bereits beim Import als Werte einfriert.
    Ein ``patch("src.core.loaders.CSVLoader")`` ersetzt nur den Namen im
    Modul-Namespace, nicht die bereits im Dict gespeicherte Referenz.
    ``patch.dict`` tauscht den Dict-Inhalt für die Dauer des Tests und stellt
    ihn danach zuverlässig wieder her – die einzig korrekte Patch-Strategie hier.
    """

    # ── Hilfsmethode ──────────────────────────────────────────────────────────

    @staticmethod
    def _mock_loader_factory(docs: list[Document] | Exception) -> MagicMock:
        """
        Erzeugt eine Mock-Loader-Klasse, die ``loader_cls(path).load()`` simuliert.

        Args:
            docs: Entweder eine Liste von Documents (Erfolg) oder eine Exception-
                  Instanz, die beim ``.load()``-Aufruf geraised wird.
        """
        instance = MagicMock()
        if isinstance(docs, Exception):
            instance.load.side_effect = docs
        else:
            instance.load.return_value = docs
        loader_cls = MagicMock(return_value=instance)
        return loader_cls

    # ── Tests ─────────────────────────────────────────────────────────────────

    def test_empty_paths_returns_empty_list(self) -> None:
        """Leere Eingabe → keine Loader-Aufrufe, leere Ausgabe."""
        assert load_documents([]) == []

    def test_unsupported_suffix_skipped(self, tmp_path: Path) -> None:
        """.txt-Datei ist nicht in der Factory → wird übersprungen, kein Crash."""
        txt = tmp_path / "readme.txt"
        txt.write_text("some text")
        result = load_documents([txt])
        assert result == []

    def test_csv_file_uses_csv_loader(self, tmp_path: Path) -> None:
        """CSVLoader wird für .csv instanziiert und .load() aufgerufen."""
        csv_path = tmp_path / "grades.csv"
        csv_path.write_bytes(b"name,score\nalice,90\n")
        doc = _make_doc("name: alice, score: 90", str(csv_path))

        MockCSV = self._mock_loader_factory([doc])

        with patch.dict(loaders_module.LOADER_BY_SUFFIX, {".csv": MockCSV}):
            result = load_documents([csv_path])

        # Loader wurde mit dem String-Pfad konstruiert
        MockCSV.assert_called_once_with(str(csv_path))
        MockCSV.return_value.load.assert_called_once()
        assert result == [doc]

    def test_pdf_file_uses_pdf_loader(self, tmp_path: Path) -> None:
        """PyPDFLoader wird für .pdf instanziiert und .load() aufgerufen."""
        pdf_path = tmp_path / "lecture.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        doc = _make_doc("Slide 1 content", str(pdf_path))

        MockPDF = self._mock_loader_factory([doc])

        with patch.dict(loaders_module.LOADER_BY_SUFFIX, {".pdf": MockPDF}):
            result = load_documents([pdf_path])

        MockPDF.assert_called_once_with(str(pdf_path))
        MockPDF.return_value.load.assert_called_once()
        assert result == [doc]

    def test_multiple_documents_from_one_file(self, tmp_path: Path) -> None:
        """Loader gibt mehrere Chunks zurück → alle landen in der Ausgabe."""
        csv_path = tmp_path / "multi.csv"
        csv_path.write_bytes(b"x")
        docs = [
            _make_doc("row 1", str(csv_path)),
            _make_doc("row 2", str(csv_path)),
            _make_doc("row 3", str(csv_path)),
        ]

        MockCSV = self._mock_loader_factory(docs)

        with patch.dict(loaders_module.LOADER_BY_SUFFIX, {".csv": MockCSV}):
            result = load_documents([csv_path])

        assert len(result) == 3
        assert result == docs

    def test_mixed_files_correct_loader_per_suffix(self, tmp_path: Path) -> None:
        """CSV-Datei → CSV-Mock, PDF-Datei → PDF-Mock, beide nie vertauscht."""
        csv_path = tmp_path / "data.csv"
        pdf_path = tmp_path / "slides.pdf"
        csv_path.write_bytes(b"x")
        pdf_path.write_bytes(b"x")

        csv_doc = _make_doc("csv content", str(csv_path))
        pdf_doc = _make_doc("pdf content", str(pdf_path))
        MockCSV = self._mock_loader_factory([csv_doc])
        MockPDF = self._mock_loader_factory([pdf_doc])

        with patch.dict(loaders_module.LOADER_BY_SUFFIX, {".csv": MockCSV, ".pdf": MockPDF}):
            result = load_documents([csv_path, pdf_path])

        MockCSV.assert_called_once_with(str(csv_path))
        MockPDF.assert_called_once_with(str(pdf_path))
        assert csv_doc in result
        assert pdf_doc in result

    def test_documents_from_multiple_files_concatenated(self, tmp_path: Path) -> None:
        """Dokumente aus allen Dateien werden zu einer einzigen Liste zusammengeführt."""
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        f1.write_bytes(b"x")
        f2.write_bytes(b"x")

        doc1 = _make_doc("from a", str(f1))
        doc2 = _make_doc("from b", str(f2))

        # Zwei separate Instanzen via side_effect auf die Klasse selbst
        MockCSV = MagicMock()
        instance1, instance2 = MagicMock(), MagicMock()
        instance1.load.return_value = [doc1]
        instance2.load.return_value = [doc2]
        MockCSV.side_effect = [instance1, instance2]

        with patch.dict(loaders_module.LOADER_BY_SUFFIX, {".csv": MockCSV}):
            result = load_documents([f1, f2])

        assert result == [doc1, doc2]

    def test_file_not_found_skipped_gracefully(self, tmp_path: Path) -> None:
        """FileNotFoundError → Datei wird übersprungen, restliche Verarbeitung läuft weiter."""
        missing  = tmp_path / "ghost.csv"
        existing = tmp_path / "real.csv"
        existing.write_bytes(b"x")
        real_doc = _make_doc("real content", str(existing))

        # Instanz 1 (missing) → FileNotFoundError; Instanz 2 (existing) → Dokument
        MockCSV = MagicMock()
        bad_instance  = MagicMock()
        good_instance = MagicMock()
        bad_instance.load.side_effect  = FileNotFoundError("not found")
        good_instance.load.return_value = [real_doc]
        MockCSV.side_effect = [bad_instance, good_instance]

        with patch.dict(loaders_module.LOADER_BY_SUFFIX, {".csv": MockCSV}):
            result = load_documents([missing, existing])

        assert result == [real_doc]

    def test_generic_exception_skipped_gracefully(self, tmp_path: Path) -> None:
        """Beliebige Exception → Datei wird übersprungen, keine Exception propagiert."""
        bad_csv = tmp_path / "broken.csv"
        bad_csv.write_bytes(b"corrupt data")

        MockCSV = self._mock_loader_factory(RuntimeError("parse error"))

        with patch.dict(loaders_module.LOADER_BY_SUFFIX, {".csv": MockCSV}):
            result = load_documents([bad_csv])

        assert result == []

    def test_error_in_one_file_does_not_abort_others(self, tmp_path: Path) -> None:
        """Exception bei Datei N bricht nicht das Laden von Datei N+1 ab."""
        bad  = tmp_path / "bad.csv"
        good = tmp_path / "good.csv"
        bad.write_bytes(b"x")
        good.write_bytes(b"x")
        good_doc = _make_doc("good content", str(good))

        MockCSV = MagicMock()
        bad_instance  = MagicMock()
        good_instance = MagicMock()
        bad_instance.load.side_effect   = ValueError("encoding error")
        good_instance.load.return_value = [good_doc]
        MockCSV.side_effect = [bad_instance, good_instance]

        with patch.dict(loaders_module.LOADER_BY_SUFFIX, {".csv": MockCSV}):
            result = load_documents([bad, good])

        assert result == [good_doc]

    def test_loader_called_with_string_path_not_path_object(self, tmp_path: Path) -> None:
        """LangChain-Loader erwarten str, kein pathlib.Path-Objekt."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_bytes(b"x")

        MockCSV = self._mock_loader_factory([])

        with patch.dict(loaders_module.LOADER_BY_SUFFIX, {".csv": MockCSV}):
            load_documents([csv_path])

        constructor_arg = MockCSV.call_args[0][0]
        assert isinstance(constructor_arg, str), (
            f"Loader wurde mit {type(constructor_arg).__name__} statt str aufgerufen"
        )

    def test_unsupported_file_does_not_instantiate_any_loader(self, tmp_path: Path) -> None:
        """.txt-Datei darf keinen Loader instanziieren."""
        txt = tmp_path / "notes.txt"
        txt.write_text("hello")

        MockCSV = self._mock_loader_factory([])
        MockPDF = self._mock_loader_factory([])

        with patch.dict(loaders_module.LOADER_BY_SUFFIX, {".csv": MockCSV, ".pdf": MockPDF}):
            load_documents([txt])

        MockCSV.assert_not_called()
        MockPDF.assert_not_called()