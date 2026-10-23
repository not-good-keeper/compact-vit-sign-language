"""Copy the 609 CISLR test rows with corpus relabelled to 'include' (see run_crosscorpus_release.sh)."""
import csv
import sys

SRC = "splits/crosscorpus__cislr-test.csv"

with open(SRC, encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    fields = reader.fieldnames
    rows = [row for row in reader if row["split"] == "test"]
assert len(rows) == 609 and all(row["corpus"] == "cislr" for row in rows), "unexpected CISLR test rows"
for row in rows:
    row["corpus"] = "include"
with open(sys.argv[1], "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {len(rows)} rows to {sys.argv[1]}")
