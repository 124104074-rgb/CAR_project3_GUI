import csv
import time
import os

if os.path.exists("live_data.csv"):
    os.remove("live_data.csv")
source_file = "arvind.csv"      # Original complete dataset
live_file = "live_data.csv"     # Real-time simulation file

# Read complete dataset
with open(source_file, "r") as f:
    rows = list(csv.reader(f))

print(f"Loaded {len(rows)-1} data rows")
# Create fresh live_data.csv with only header

with open(live_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(rows[0])

print("Live file created")
for row in rows[1:]:

    with open(live_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print("Written:", row)

    # Philips monitor updates every 5 sec
    time.sleep(0.05)
