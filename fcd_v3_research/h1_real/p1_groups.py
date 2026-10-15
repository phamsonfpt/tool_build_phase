"""P1-groups — metadata only. Rides (rideID), blocks and super-blocks, and the data roles of H1 v1.4.

Needs cache/info_meta.json (p1_info.py) and cache/p1_units.json (p1_metadata.py).
Writes cache/groups.json and results/p1_info_report.json. No box label is read.

Registered definitions (H1 v1.4, Section VII.P1):
  ride        = rideID of the Info file (equals the 8-hex video-name prefix one-to-one)
  position    = first GPS fix and startTime of the ride's earliest clip
  local time  = zoneinfo of the region (lon/lat boxes below); 'UTC' if outside every box
  grid        = equirectangular km grid with a fixed reference latitude per region, common origin
  block       = 1 km x 1 km cell x 2-hour local window ; super-block = 5 km x 5 km cell x local calendar day
"""
import collections
import datetime
import json
import math
import os
from zoneinfo import ZoneInfo

from h1common import CACHE, RESULTS, write_json

REGIONS = [("America/Los_Angeles", (-130, -100), (20, 60), 37.8),
           ("America/New_York", (-90, -60), (20, 60), 40.7),
           ("Asia/Jerusalem", (29, 40), (27, 36), 32.0)]


def region(lat, lon):
    for tz, (lo0, lo1), (la0, la1), ref in REGIONS:
        if lo0 < lon < lo1 and la0 < lat < la1:
            return tz, ref
    return "UTC", lat


def groups_of(v):
    tz, ref = region(v["lat"], v["lon"])
    x = 111.32 * math.cos(math.radians(ref)) * v["lon"]
    y = 110.574 * v["lat"]
    t = datetime.datetime.fromtimestamp(v["t0"] / 1000, ZoneInfo(tz))
    day = t.date().isoformat()
    sb = f"{tz}|{math.floor(x / 5)}|{math.floor(y / 5)}|{day}"
    bl = f"{tz}|{math.floor(x)}|{math.floor(y)}|{day}|{t.hour // 2}"
    return tz, bl, sb


def m_tilde(counter):
    return sum(c * c for c in counter.values()) / max(sum(counter.values()), 1)


def main():
    info = json.load(open(os.path.join(CACHE, "info_meta.json")))
    units = json.load(open(os.path.join(CACHE, "p1_units.json")))
    rides = collections.defaultdict(list)                       # rideID -> videos
    for name, v in info.items():
        if v["ride"]:
            rides[v["ride"]].append(name)
    prefix_of = {}
    for r, ns in rides.items():
        prefix_of[r] = {n.split("-")[0] for n in ns}
    one_to_one = all(len(p) == 1 for p in prefix_of.values()) and \
        len({next(iter(p)) for p in prefix_of.values()}) == len(prefix_of)
    ride_of_prefix = {next(iter(p)): r for r, p in prefix_of.items()}

    def ride_groups(prefix):
        r = ride_of_prefix.get(prefix)
        ns = [n for n in rides.get(r, []) if info[n]["t0"] and info[n]["lat"] is not None]
        if not ns:
            return None
        first = min(ns, key=lambda n: info[n]["t0"])
        return groups_of(info[first])

    gaps = []
    for r, ns in rides.items():
        ts = sorted((info[n]["t0"], info[n]["t1"] or info[n]["t0"]) for n in ns if info[n]["t0"])
        gaps += [(b[0] - a[1]) / 60000 for a, b in zip(ts, ts[1:])]
    gaps.sort()

    split_rides = collections.defaultdict(set)
    for name, v in info.items():
        if v["ride"]:
            split_rides[v["split"]].add(name.split("-")[0])
    pools = {"V": sorted(units["val"]), "M": sorted(units["meta"]),
             "T": sorted(p for p in split_rides["test"] if p not in units["meta"])}
    pools["Train"] = sorted(split_rides["train"])
    G, report = {}, {"ride_id": {"videos_with_rideID": sum(len(v) for v in rides.values()), "rideIDs": len(rides),
                                 "prefix_rideID_one_to_one": one_to_one,
                                 "within_ride_gap_minutes": {q: round(gaps[int(q * (len(gaps) - 1))], 1)
                                                             for q in (0.1, 0.5, 0.9, 0.99)},
                                 "within_ride_gap_max_minutes": round(gaps[-1], 1)},
                     "timezones": dict(collections.Counter(region(v["lat"], v["lon"])[0] for v in info.values()
                                                           if v["lat"] is not None)),
                     "unparseable_info_files": sum(not v.get("parsed", True) for v in info.values()),
                     "videos_without_gps": sum(v["lat"] is None for v in info.values())}
    for pool, prefixes in pools.items():
        g = {p: ride_groups(p) for p in prefixes}
        ok = {p: x for p, x in g.items() if x}
        sb = collections.Counter(x[2] for x in ok.values()); bl = collections.Counter(x[1] for x in ok.values())
        G[pool] = {p: {"block": x[1], "super_block": x[2], "tz": x[0]} for p, x in ok.items()}
        report[f"pool_{pool}"] = {"rides": len(prefixes), "rides_with_metadata": len(ok),
                                  "super_blocks": len(sb), "m_tilde_super_block": m_tilde(sb),
                                  "max_super_block": max(sb.values()), "blocks": len(bl),
                                  "m_tilde_block": m_tilde(bl), "max_block": max(bl.values())}
    sb_train = {x["super_block"] for x in G["Train"].values()}
    sb_v = {x["super_block"] for x in G["V"].values()}
    sb_m = {x["super_block"] for x in G["M"].values()}
    report["overlap"] = {
        "V_rides_sharing_super_block_with_detector_train": sum(x["super_block"] in sb_train for x in G["V"].values()) / len(G["V"]),
        "T_rides_sharing_super_block_with_detector_train": sum(x["super_block"] in sb_train for x in G["T"].values()) / len(G["T"]),
        "T_rides_sharing_super_block_with_V": sum(x["super_block"] in sb_v for x in G["T"].values()) / len(G["T"]),
        "T_rides_in_super_blocks_new_to_V_and_M": sum(x["super_block"] not in sb_v | sb_m for x in G["T"].values()),
        "T_rides_in_super_blocks_new_to_V_M_and_train": sum(x["super_block"] not in sb_v | sb_m | sb_train for x in G["T"].values()),
    }
    n_t = report["pool_T"]["rides_with_metadata"]
    report["feasibility_T_N1500_K3"] = n_t >= 3 * 1500 + 3 * report["pool_T"]["max_super_block"]
    write_json(os.path.join(CACHE, "groups.json"), G)
    write_json(os.path.join(RESULTS, "p1_info_report.json"), report)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
