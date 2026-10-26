#!/usr/bin/env python3
"""Repair the Drive copy of notebooks/3_full_paper.ipynb after an accidental
keystroke edit (2026-10-25): drop a stray first cell, restore cells that were
deleted (taken from the repo copy, outputs cleared) in front of a given anchor
cell, and append repo-only cells of the trailing section.  Writes a timestamped
backup next to the Drive notebook before overwriting it.  No data files touched.

Usage (Colab Terminal, Drive mounted):
  python scripts/repair_colab_notebook.py --drive-nb "<path to Drive .ipynb>" \
      --repo-nb notebooks/3_full_paper.ipynb --stray-prefix "il -3 /content/de_addons" \
      --restore "# (h8h)" "# (h8i)" --before "# (h8j)" --append-from "## 9. 决策错误率"
"""
import argparse, copy, json, shutil, time, uuid


def src(c):
    return "".join(c.get("source", []))


def fresh(c):
    c = copy.deepcopy(c)
    if c.get("cell_type") == "code":
        c["outputs"] = []
        c["execution_count"] = None
    c["metadata"] = {"id": uuid.uuid4().hex[:12]}
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive-nb", required=True)
    ap.add_argument("--repo-nb", required=True)
    ap.add_argument("--stray-prefix", default=None,
                    help="drop the FIRST cell iff its source starts with this")
    ap.add_argument("--restore", nargs="*", default=[],
                    help="source prefixes of repo cells to re-insert (in order)")
    ap.add_argument("--before", default=None,
                    help="source prefix of the Drive cell before which to insert")
    ap.add_argument("--append-from", default=None,
                    help="append repo cells from this heading on, if absent in Drive")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    drive = json.load(open(a.drive_nb, encoding="utf-8"))
    repo = json.load(open(a.repo_nb, encoding="utf-8"))
    cells = drive["cells"]
    n0 = len(cells)
    log = []
    if a.stray_prefix and cells and src(cells[0]).lstrip().startswith(a.stray_prefix):
        log.append(f"dropped stray first cell: {src(cells[0])[:60]!r}")
        cells = cells[1:]
    if a.restore:
        have = {src(c).lstrip() for c in cells}
        todo = []
        for pref in a.restore:
            rc = [c for c in repo["cells"] if src(c).lstrip().startswith(pref)]
            if not rc:
                raise SystemExit(f"ERROR: repo has no cell starting with {pref!r}")
            if any(h.startswith(pref) for h in have):
                log.append(f"already present, skipped: {pref!r}")
                continue
            todo.append(fresh(rc[0]))
        if todo:
            pos = len(cells)
            if a.before:
                hits = [i for i, c in enumerate(cells) if src(c).lstrip().startswith(a.before)]
                if not hits:
                    raise SystemExit(f"ERROR: Drive copy has no cell starting with {a.before!r}")
                pos = hits[0]
            cells[pos:pos] = todo
            log.append(f"restored {len(todo)} cell(s) at index {pos}: "
                       + ", ".join(src(c)[:40].strip() for c in todo))
    if a.append_from:
        ks = [i for i, c in enumerate(repo["cells"]) if src(c).lstrip().startswith(a.append_from)]
        if ks:
            have = {src(c) for c in cells}
            add = [fresh(c) for c in repo["cells"][ks[0]:] if src(c) not in have]
            cells.extend(add)
            log.append(f"appended {len(add)} repo-only cell(s) from {a.append_from!r}")
    drive["cells"] = cells
    print(f"cells {n0} -> {len(cells)}")
    for l in log:
        print("  -", l)
    if a.dry_run:
        print("dry run; nothing written")
        return
    bak = a.drive_nb + f".bak-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(a.drive_nb, bak)
    with open(a.drive_nb, "w", encoding="utf-8") as f:
        json.dump(drive, f, ensure_ascii=False, indent=1)
    print(f"backup: {bak}\nwrote: {a.drive_nb}")


if __name__ == "__main__":
    main()
