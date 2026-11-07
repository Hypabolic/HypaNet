"""Print the header lines (and hunk lines) of the first N events of given windows. Usage: ./py heads.py N ID [ID...]"""
import sys

import v05_load


def main():
    n = int(sys.argv[1])
    ids = sys.argv[2:]
    for w in v05_load.windows("all"):
        if w["id"] in ids:
            print(f"=== {w['id']} {w['oracle_label']} t={w['t']} start={w['start']}")
            for i, b in enumerate(w["blocks"][:n]):
                lines = b.split("\n")
                print(f" {i:3d} {lines[0].split(' ', 1)[1][:90]}")
                for ln in lines[1:]:
                    print(f"       {ln.strip()[:80]}")


if __name__ == "__main__":
    main()
