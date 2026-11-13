import tempfile
import unittest
from pathlib import Path

from practice.analysis import filter_records, read_records, summarize


class AnalysisTests(unittest.TestCase):
    def make_csv(self, content):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "test.csv"
        path.write_text(content, encoding="utf-8")
        return path

    def test_valid_records_sorted(self):
        p = self.make_csv(
            "timestamp_utc,station,temperature_c,voltage_v\n"
            "2026-01-02T00:00:00+00:00,Bench-B,32.5,3.4\n"
            "2026-01-01T00:00:00+00:00,Bench-A,30,3.2\n"
        )
        rows = read_records(p)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["station"], "Bench-A")

    def test_missing_column(self):
        p = self.make_csv("timestamp_utc,station,temperature_c\n2026-01-01,Bench-A,30\n")
        with self.assertRaisesRegex(ValueError, "voltage_v"):
            read_records(p)

    def test_bad_temperature(self):
        p = self.make_csv("timestamp_utc,station,temperature_c,voltage_v\n2026-01-01,Bench-A,not_a_number,3.3\n")
        with self.assertRaisesRegex(ValueError, "row 2"):
            read_records(p)

    def test_reject_nan(self):
        p = self.make_csv("timestamp_utc,station,temperature_c,voltage_v\n2026-01-01,Bench-A,nan,3.3\n")
        with self.assertRaisesRegex(ValueError, "row 2"):
            read_records(p)

    def test_filter_stations(self):
        rows = [{"station": "A"}, {"station": "B"}, {"station": "A"}]
        self.assertEqual(len(filter_records(rows, ["A"])), 2)
        self.assertEqual(len(filter_records(rows, [])), 0)

    def test_summary(self):
        rows = [
            {"station": "A", "temperature_c": 20.0, "voltage_v": 3.0},
            {"station": "B", "temperature_c": 30.0, "voltage_v": 3.4},
        ]
        summary = summarize(rows)
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["stations"], 2)
        self.assertAlmostEqual(summary["mean_temperature_c"], 25.0)
        self.assertAlmostEqual(summary["mean_voltage_v"], 3.2)

    def test_empty_summary(self):
        self.assertIsNone(summarize([])["mean_temperature_c"])
        self.assertEqual(summarize([])["count"], 0)


if __name__ == "__main__":
    unittest.main()
