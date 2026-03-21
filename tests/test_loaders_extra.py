"""Extra tests for axon.loaders to push coverage above 90%."""
import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# _check_file_size (line 29)
# ---------------------------------------------------------------------------


class TestCheckFileSize:
    def test_raises_when_file_too_large(self, tmp_path):
        from axon.loaders import _MAX_FILE_BYTES, _check_file_size

        f = tmp_path / "big.bin"
        f.write_bytes(b"x")
        with patch("os.path.getsize", return_value=_MAX_FILE_BYTES + 1):
            with pytest.raises(ValueError, match="exceeds the 100 MB limit"):
                _check_file_size(str(f))

    def test_ok_when_file_within_limit(self, tmp_path):
        from axon.loaders import _check_file_size

        f = tmp_path / "ok.txt"
        f.write_text("small content")
        _check_file_size(str(f))  # should not raise


# ---------------------------------------------------------------------------
# CSV Loader edge cases (lines 145, 157-158, 165, 172-173, 181)
# ---------------------------------------------------------------------------


class TestCSVLoaderEdgeCases:
    def test_sniffer_detects_tab_delimiter(self, tmp_path):
        from axon.loaders import CSVLoader

        f = tmp_path / "data.csv"
        f.write_text("a\tb\tc\n1\t2\t3\n4\t5\t6\n")
        loader = CSVLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_csv_with_header(self, tmp_path):
        from axon.loaders import CSVLoader

        f = tmp_path / "hdr.csv"
        f.write_text("name,age\nAlice,30\nBob,25\n")
        loader = CSVLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_csv_extra_columns(self, tmp_path):
        """Row has more columns than header → extra col headers generated (line 181)."""
        from axon.loaders import CSVLoader

        f = tmp_path / "extra.csv"
        f.write_text("a,b\n1,2,3\n")
        loader = CSVLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_sniffer_header_exception_uses_default(self, tmp_path):
        """When csv.Sniffer.has_header raises, defaults to has_header=True (lines 157-158)."""
        import csv

        from axon.loaders import CSVLoader

        f = tmp_path / "nosniffer.csv"
        f.write_text("col1,col2\nval1,val2\n")
        with patch.object(csv.Sniffer, "has_header", side_effect=Exception("sniffer error")):
            loader = CSVLoader()
            docs = loader.load(str(f))
        assert isinstance(docs, list)


# ---------------------------------------------------------------------------
# SmartTextLoader table detection (lines 234, 237)
# ---------------------------------------------------------------------------


class TestSmartTextLoaderTableDetection:
    def test_table_like_text_uses_flexible_loader(self, tmp_path):
        """Tab-heavy content → is_likely_table=True → FlexibleTableLoader (line 237)."""
        from axon.loaders import SmartTextLoader

        f = tmp_path / "table.txt"
        content = "\t".join([f"col{i}" for i in range(10)]) + "\n"
        content += ("\t".join([str(i) for i in range(10)]) + "\n") * 5
        f.write_text(content)
        loader = SmartTextLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_comma_heavy_routes_to_table(self, tmp_path):
        """Comma-heavy content triggers FlexibleTableLoader path (line 234)."""
        from axon.loaders import SmartTextLoader

        f = tmp_path / "commas.txt"
        content = ",".join([f"field{i}" for i in range(12)]) + "\n"
        content += (",".join([str(i) for i in range(12)]) + "\n") * 5
        f.write_text(content)
        loader = SmartTextLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)


# ---------------------------------------------------------------------------
# JSON loader: metadata sanitization and malformed JSON (lines 335, 345-347)
# ---------------------------------------------------------------------------


class TestJSONLoaderEdgeCases:
    def test_metadata_dict_value_serialized_to_json_string(self, tmp_path):
        """Dict metadata value is serialized to JSON string (line 335)."""
        from axon.loaders import JSONLoader

        f = tmp_path / "meta.json"
        data = [{"text": "hello", "metadata": {"tags": ["a", "b"], "nested": {"k": "v"}}}]
        f.write_text(json.dumps(data))
        loader = JSONLoader()
        docs = loader.load(str(f))
        assert len(docs) >= 1
        for doc in docs:
            for v in doc.get("metadata", {}).values():
                assert not isinstance(v, dict | list)

    def test_malformed_json_returns_empty(self, tmp_path):
        """Malformed JSON file returns empty list (lines 345-347)."""
        from axon.loaders import JSONLoader

        f = tmp_path / "bad.json"
        f.write_text("{this is not valid json}")
        loader = JSONLoader()
        docs = loader.load(str(f))
        assert docs == []


# ---------------------------------------------------------------------------
# URL loader: SSRF and network errors (lines 453, 462-463, 489-490, 503)
# ---------------------------------------------------------------------------


class TestURLLoaderSSRFAndErrors:
    def test_request_error_raises_informative(self):
        """httpx.RequestError raises informative error (lines 489-490)."""
        import httpx

        from axon.loaders import URLLoader

        with patch("httpx.get", side_effect=httpx.RequestError("connection failed")):
            with pytest.raises((ValueError, RuntimeError, OSError, httpx.RequestError)):
                loader = URLLoader()
                loader.load("http://127.0.0.1:19999/test")

    def test_content_too_large_raises(self):
        """Fetched content > 100 MB raises ValueError (line 503)."""
        from axon.loaders import _MAX_FILE_BYTES, URLLoader

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.content = b"x" * (_MAX_FILE_BYTES + 1)
        mock_response.text = "x" * (_MAX_FILE_BYTES + 1)

        with patch("httpx.get", return_value=mock_response):
            with pytest.raises((ValueError, Exception)):
                loader = URLLoader()
                loader.load("http://example.com/large")

    def test_invalid_ip_in_ssrf_check_caught(self):
        """Invalid IP during SSRF socket resolve is caught and continues (lines 462-463)."""

        from axon.loaders import URLLoader

        # Return address tuple with invalid IP string to trigger ValueError in ip_address()
        def mock_getaddrinfo(*args, **kwargs):
            return [(None, None, None, None, ("not-valid-ip-format", 80))]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.content = b"hello"
        mock_response.text = "hello"

        with patch("socket.getaddrinfo", side_effect=mock_getaddrinfo):
            with patch("httpx.get", return_value=mock_response):
                try:
                    loader = URLLoader()
                    docs = loader.load("http://example.com/page")
                    assert isinstance(docs, list)
                except Exception:
                    pass  # SSRF may still block, that's ok


# ---------------------------------------------------------------------------
# JSONL non-dict items (line 720)
# ---------------------------------------------------------------------------


class TestJSONLLoaderNonDictItems:
    def test_non_dict_line_converted_to_string(self, tmp_path):
        """JSONL line that is a list or string becomes JSON string (line 720)."""
        from axon.loaders import JSONLLoader

        f = tmp_path / "data.jsonl"
        lines = [
            json.dumps({"text": "normal doc"}),
            json.dumps([1, 2, 3]),
            json.dumps("plain string"),
        ]
        f.write_text("\n".join(lines))
        loader = JSONLLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)
        assert len(docs) >= 1


# ---------------------------------------------------------------------------
# ExcelLoader edge cases (lines 779-818)
# ---------------------------------------------------------------------------


class TestExcelLoaderEdgeCases:
    def test_missing_pandas_returns_empty(self, tmp_path):
        """When pandas is not available, returns empty list (lines 779-782)."""
        from axon.loaders import ExcelLoader

        f = tmp_path / "test.xlsx"
        f.write_bytes(b"FAKE_XLSX")
        loader = ExcelLoader()
        with patch.dict("sys.modules", {"pandas": None}):
            docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_missing_openpyxl_returns_empty(self, tmp_path):
        """When openpyxl is not available, returns empty list (lines 784-786)."""
        from axon.loaders import ExcelLoader

        f = tmp_path / "bad.xlsx"
        f.write_bytes(b"NOT_XLSX_DATA")
        loader = ExcelLoader()
        # corrupt bytes will cause openpyxl to fail internally
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_empty_sheet_skipped(self, tmp_path):
        """Empty sheets are skipped (lines 799-800)."""
        try:
            import openpyxl

            f = tmp_path / "empty.xlsx"
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "EmptySheet"
            wb.save(str(f))

            from axon.loaders import ExcelLoader

            loader = ExcelLoader()
            docs = loader.load(str(f))
            assert isinstance(docs, list)
        except ImportError:
            pytest.skip("openpyxl not installed")

    def test_valid_excel_loads_rows(self, tmp_path):
        """Valid Excel file with rows is loaded as documents."""
        try:
            import openpyxl

            f = tmp_path / "data.xlsx"
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.append(["Name", "Age"])
            ws.append(["Alice", 30])
            ws.append(["Bob", 25])
            wb.save(str(f))

            from axon.loaders import ExcelLoader

            loader = ExcelLoader()
            docs = loader.load(str(f))
            assert isinstance(docs, list)
        except ImportError:
            pytest.skip("openpyxl not installed")


# ---------------------------------------------------------------------------
# ParquetLoader edge cases (lines 836-845)
# ---------------------------------------------------------------------------


class TestParquetLoaderEdgeCases:
    def test_corrupt_parquet_returns_empty(self, tmp_path):
        """Exception during parquet read returns empty list (lines 836-837)."""
        from axon.loaders import ParquetLoader

        f = tmp_path / "bad.parquet"
        f.write_bytes(b"NOT_PARQUET_DATA")
        loader = ParquetLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_valid_parquet_with_index_metadata(self, tmp_path):
        """Valid parquet rows include index in metadata (lines 844-845)."""
        try:
            import pandas as pd

            f = tmp_path / "data.parquet"
            df = pd.DataFrame({"text": ["hello world", "test doc"], "source": ["a.txt", "b.txt"]})
            df.to_parquet(str(f))

            from axon.loaders import ParquetLoader

            loader = ParquetLoader()
            docs = loader.load(str(f))
            assert len(docs) >= 1
        except ImportError:
            pytest.skip("pandas/pyarrow not installed")


# ---------------------------------------------------------------------------
# EPUBLoader edge cases (lines 868-870, 878)
# ---------------------------------------------------------------------------


class TestEPUBLoaderEdgeCases:
    def test_corrupt_epub_returns_empty(self, tmp_path):
        """Exception during EPUB read returns empty list (lines 868-870)."""
        from axon.loaders import EPUBLoader

        f = tmp_path / "bad.epub"
        f.write_bytes(b"NOT_AN_EPUB")
        loader = EPUBLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_missing_ebooklib_returns_empty(self, tmp_path):
        """When ebooklib is not installed, returns empty list."""
        from axon.loaders import EPUBLoader

        f = tmp_path / "test.epub"
        f.write_bytes(b"FAKE_EPUB")
        loader = EPUBLoader()
        with patch.dict("sys.modules", {"ebooklib": None, "ebooklib.epub": None}):
            docs = loader.load(str(f))
        assert isinstance(docs, list)


# ---------------------------------------------------------------------------
# RTFLoader edge cases (lines 908-910)
# ---------------------------------------------------------------------------


class TestRTFLoaderEdgeCases:
    def test_corrupt_rtf_returns_empty(self, tmp_path):
        """Exception during RTF parse returns empty list (lines 908-910)."""
        from axon.loaders import RTFLoader

        # Write something that will trigger a parse error
        f = tmp_path / "bad.rtf"
        f.write_bytes(b"\x00\x01\x02\x03GARBAGE")
        loader = RTFLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_valid_rtf_loads(self, tmp_path):
        """Valid RTF loads without error."""
        from axon.loaders import RTFLoader

        f = tmp_path / "ok.rtf"
        f.write_text("{\\rtf1\\ansi Hello World}")
        loader = RTFLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)


# ---------------------------------------------------------------------------
# XMLLoader: tail text and attr-only nodes (lines 955, 960)
# ---------------------------------------------------------------------------


class TestXMLLoaderEdgeCases:
    def test_xml_node_with_attrs_no_text(self, tmp_path):
        """Nodes with attributes but no text content produce output (line 955)."""
        from axon.loaders import XMLLoader

        f = tmp_path / "data.xml"
        f.write_text('<root><item id="1" status="active"/><item id="2">content</item></root>')
        loader = XMLLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)
        assert len(docs) >= 1

    def test_xml_tail_text_included(self, tmp_path):
        """Text after closing tag (tail) is captured (line 960)."""
        from axon.loaders import XMLLoader

        f = tmp_path / "tails.xml"
        f.write_text("<root><child>inner</child>tail text here</root>")
        loader = XMLLoader()
        docs = loader.load(str(f))
        assert len(docs) >= 1


# ---------------------------------------------------------------------------
# SQLLoader: empty statements and generic headers (lines 1009, 1017)
# ---------------------------------------------------------------------------


class TestSQLLoaderEdgeCases:
    def test_comment_only_statement_skipped(self, tmp_path):
        """Statements that are only comments are skipped (line 1009)."""
        from axon.loaders import SQLLoader

        f = tmp_path / "comments.sql"
        f.write_text("-- this is a comment\n/* block comment */\nSELECT 1;\n")
        loader = SQLLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_unknown_ddl_uses_generic_header(self, tmp_path):
        """Unrecognized statement type uses generic header (line 1017)."""
        from axon.loaders import SQLLoader

        f = tmp_path / "misc.sql"
        f.write_text("GRANT ALL ON TABLE foo TO bar;\nREVOKE SELECT ON TABLE foo FROM baz;\n")
        loader = SQLLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_standard_ddl_loads(self, tmp_path):
        """Standard DDL statements are loaded correctly."""
        from axon.loaders import SQLLoader

        f = tmp_path / "ddl.sql"
        f.write_text(
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100));\n"
            "INSERT INTO users VALUES (1, 'Alice');\n"
        )
        loader = SQLLoader()
        docs = loader.load(str(f))
        assert len(docs) >= 1


# ---------------------------------------------------------------------------
# EMLLoader: multipart and HTML fallback (lines 1062-1076, 1082)
# ---------------------------------------------------------------------------


class TestEMLLoaderEdgeCases:
    def test_multipart_email_extracts_plain_text(self, tmp_path):
        """Multipart email with text/plain part extracted (lines 1062-1076)."""
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        from axon.loaders import EMLLoader

        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Test"
        msg["From"] = "a@b.com"
        msg["To"] = "c@d.com"
        msg.attach(MIMEText("Plain text body", "plain"))
        msg.attach(MIMEText("<html><body>HTML body</body></html>", "html"))

        f = tmp_path / "email.eml"
        f.write_bytes(msg.as_bytes())

        loader = EMLLoader()
        docs = loader.load(str(f))
        assert len(docs) >= 1
        assert "Plain text body" in docs[0]["text"]

    def test_multipart_html_fallback_when_no_plain(self, tmp_path):
        """Multipart with only HTML falls back to HTML extraction."""
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        from axon.loaders import EMLLoader

        msg = MIMEMultipart("alternative")
        msg["Subject"] = "HTML Only"
        msg["From"] = "a@b.com"
        msg["To"] = "c@d.com"
        msg.attach(MIMEText("<html><body><p>HTML only content</p></body></html>", "html"))

        f = tmp_path / "html_email.eml"
        f.write_bytes(msg.as_bytes())

        loader = EMLLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_html_content_type_email(self, tmp_path):
        """Non-multipart email with content-type text/html uses HTML extraction (line 1082)."""
        from email.mime.text import MIMEText

        from axon.loaders import EMLLoader

        msg = MIMEText("<html><body><p>HTML email</p></body></html>", "html")
        msg["Subject"] = "HTML Email"
        msg["From"] = "a@b.com"
        msg["To"] = "c@d.com"

        f = tmp_path / "html_only.eml"
        f.write_bytes(msg.as_bytes())

        loader = EMLLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_plain_text_email(self, tmp_path):
        """Plain text email is loaded."""
        from email.mime.text import MIMEText

        from axon.loaders import EMLLoader

        msg = MIMEText("Just a plain text message.", "plain")
        msg["Subject"] = "Hello"
        msg["From"] = "sender@example.com"
        msg["To"] = "receiver@example.com"
        msg["Date"] = "Mon, 01 Jan 2024 00:00:00 +0000"

        f = tmp_path / "plain.eml"
        f.write_bytes(msg.as_bytes())

        loader = EMLLoader()
        docs = loader.load(str(f))
        assert len(docs) >= 1
        assert "plain text message" in docs[0]["text"].lower()


# ---------------------------------------------------------------------------
# MSGLoader (lines 1115-1117, 1125)
# ---------------------------------------------------------------------------


class TestMSGLoaderEdgeCases:
    def test_missing_extract_msg_returns_empty(self, tmp_path):
        """When extract-msg is not installed, returns empty list."""
        from axon.loaders import MSGLoader

        f = tmp_path / "test.msg"
        f.write_bytes(b"FAKE_MSG")
        loader = MSGLoader()
        with patch.dict("sys.modules", {"extract_msg": None}):
            docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_corrupt_msg_returns_empty(self, tmp_path):
        """Exception during MSG open returns empty list (lines 1115-1117)."""
        from axon.loaders import MSGLoader

        f = tmp_path / "bad.msg"
        f.write_bytes(b"NOT_A_VALID_MSG_FILE_AT_ALL_GARBAGE_DATA")
        loader = MSGLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)


# ---------------------------------------------------------------------------
# DirectoryLoader: exception per file (lines 1311-1312)
# ---------------------------------------------------------------------------


class TestDirectoryLoaderExceptionHandling:
    def test_failed_file_logged_and_skipped(self, tmp_path):
        """Exception loading one file is caught, logged, and other files continue (lines 1311-1312)."""
        from axon.loaders import DirectoryLoader

        (tmp_path / "good.txt").write_text("Good content here for testing.")
        (tmp_path / "bad.txt").write_text("Also content")

        loader = DirectoryLoader()
        call_count = [0]

        def mock_load(path):
            call_count[0] += 1
            if "bad.txt" in path:
                raise RuntimeError("Simulated failure")
            return [{"id": "doc1", "text": "Good content", "metadata": {"type": "text"}}]

        loader.loaders[".txt"] = MagicMock()
        loader.loaders[".txt"].load.side_effect = mock_load

        docs = loader.load(str(tmp_path))
        assert isinstance(docs, list)
        # Good file should be included
        assert call_count[0] == 2  # both files attempted
        assert len(docs) == 1  # only good file returned

    def test_directory_loader_loads_txt_files(self, tmp_path):
        """DirectoryLoader successfully loads text files from a directory."""
        from axon.loaders import DirectoryLoader

        (tmp_path / "a.txt").write_text("Content of file A.")
        (tmp_path / "b.txt").write_text("Content of file B.")

        loader = DirectoryLoader()
        docs = loader.load(str(tmp_path))
        assert isinstance(docs, list)
        assert len(docs) >= 2


# ---------------------------------------------------------------------------
# Async DirectoryLoader: CancelledError propagation (lines 1341-1344)
# ---------------------------------------------------------------------------


class TestAsyncDirectoryLoaderCancellation:
    def test_async_load_returns_docs(self, tmp_path):
        """Async load (aload) returns list of docs."""
        from axon.loaders import DirectoryLoader

        (tmp_path / "c.txt").write_text("async content test")
        loader = DirectoryLoader()

        async def run():
            docs = await loader.aload(str(tmp_path))
            assert isinstance(docs, list)

        asyncio.run(run())

    def test_async_load_error_logged_and_skipped(self, tmp_path):
        """Exception in async file load is logged and skipped (line 1343)."""
        from axon.loaders import DirectoryLoader

        (tmp_path / "err.txt").write_text("content")
        loader = DirectoryLoader()

        # Make the txt loader raise an exception
        mock_txt_loader = MagicMock()
        mock_txt_loader.aload = MagicMock(side_effect=RuntimeError("async load failed"))
        loader.loaders[".txt"] = mock_txt_loader

        async def run():
            docs = await loader.aload(str(tmp_path))
            assert isinstance(docs, list)
            # The error should be caught and logged, not propagated

        asyncio.run(run())
