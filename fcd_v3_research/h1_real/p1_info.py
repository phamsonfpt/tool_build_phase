"""P1-Info — metadata only: rideID, start/end time and first GPS fix of every 100K video, from bdd100k_info.zip.

usage: python p1_info.py <path to bdd100k_info.zip>
Writes cache/info_meta.json  {video_name: {split, ride, t0, t1, lat, lon, n_loc, timelapse}}
No box label is read.
"""
import json
import os
import sys
import zipfile

from h1common import CACHE, write_json


def first_fix(d):
    for key in ("locations", "gps"):
        for p in d.get(key) or []:
            lat, lon = p.get("latitude"), p.get("longitude")
            if lat is not None and lon is not None and not (lat == 0 and lon == 0):
                return float(lat), float(lon), len(d.get(key))
    return None, None, 0


def main(zpath):
    z = zipfile.ZipFile(zpath)
    out = {}
    for name in z.namelist():
        if not name.endswith(".json"):
            continue
        parts = name.split("/")                      # 100k/<split>/<video>.json
        raw = z.read(name)
        try:
            d = json.loads(raw)
        except ValueError:                           # empty or truncated info file -> metadata missing
            d = {}
        lat, lon, n = first_fix(d)
        out[parts[-1][:-5]] = {"split": parts[-2], "ride": d.get("rideID"), "t0": d.get("startTime"),
                               "t1": d.get("endTime"), "lat": lat, "lon": lon, "n_loc": n,
                               "timelapse": bool(d.get("timelapse")), "parsed": bool(d)}
    write_json(os.path.join(CACHE, "info_meta.json"), out)
    print("unparseable", sum(not v["parsed"] for v in out.values()), "videos", len(out), "with GPS", sum(v["lat"] is not None for v in out.values()),
          "with startTime", sum(v["t0"] is not None for v in out.values()))


if __name__ == "__main__":
    main(sys.argv[1])
