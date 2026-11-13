"""Generate fictional electronics-bench telemetry for Streamlit practice."""
import csv
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "demo_lab_measurements.csv"


def main():
    OUT.parent.mkdir(exist_ok=True)
    start = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("timestamp_utc", "station", "temperature_c", "voltage_v"))
        for hour in range(48):
            for station_index, station in enumerate(("Bench-A", "Bench-B")):
                timestamp = start + timedelta(hours=hour)
                # Arbitrary smooth patterns for visualization practice ONLY.
                temperature = 29 + station_index * 3 + 4 * math.sin(hour / 5)
                voltage = 3.30 + station_index * 0.04 + 0.025 * math.cos(hour / 7)
                writer.writerow((timestamp.isoformat(), station, f"{temperature:.3f}", f"{voltage:.4f}"))
    print(f"Wrote 96 fictional sensor readings to {OUT}")


if __name__ == "__main__":
    main()
