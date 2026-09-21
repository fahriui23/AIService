"""Generate deterministic CSV fixtures without storing large generated files in git."""
import argparse
import csv
from pathlib import Path


SAMPLES = (
    "Bandung no123 aplikasi ini sangat bagus",
    "Surabaya no213 pengirimannya sangat lambat",
    "no312 saya berasal dari Malang dan aplikasinya bagus",
    "no145 saya tinggal di Kota Batu",
    "no245 produk ini keras seperti batu",
    "no345 cabang Solo pelayanannya lambat",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int, choices=(100, 1000, 5000, 10000), default=1000)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["review"])
        for index in range(args.rows):
            writer.writerow([SAMPLES[index % len(SAMPLES)]])
    print(f"Generated {args.rows} rows: {args.output}")


if __name__ == "__main__":
    main()
