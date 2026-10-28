#!/usr/bin/env python3
"""Terrain clearance report for company routes.

Computes, from the open SRTM 1-arc-second DEM (AWS Terrain Tiles, "skadi"):
  * highest terrain inside a +/-10 NM corridor along each route/leg
  * Route MORA estimate per leg (Jeppesen rule: +1000 ft if terrain <= 5000 ft,
    +2000 ft above that, rounded up to the next 100 ft)
  * Grid MORA estimate for every 1x1 degree cell (same rule, whole cell)
  * MSA estimate within 25 NM of each airport (+1000 ft, rounded up to 100 ft),
    omnidirectional and per true quadrant
and renders shaded-relief overview maps, close-ups of the critical terrain,
vertical profiles and MSA diagrams into an HTML report.

TERRAIN ONLY. Man-made obstacles are not in the DEM. Values are estimates for
cross-checking and are NOT a substitute for published Jeppesen/AIP values.

Usage:
  python3 tools/terrain_report.py --routes routes.csv --out output/terrain_report
routes.csv columns: route_id,description,waypoints
  waypoints = space separated tokens; an ICAO airport code (e.g. HECA) or a
  custom fix written NAME/LAT/LON in decimal degrees (e.g. KATAB/28.12/33.45).
"""
import argparse, concurrent.futures as cf, csv, datetime, gzip, html, json, math, os, urllib.error, urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

M2FT = 3.28084
NM = 1.852            # km
R_EARTH = 6371.0088   # km
N = 1200              # 3 arc-second cells per degree (block-max of the 1" data)
CORRIDOR_NM = 10.0
MSA_NM = 25.0
TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{d}/{n}.hgt.gz"
AIRPORTS_URL = "https://raw.githubusercontent.com/davidmegginson/ourairports-data/main/airports.csv"

CMAP = LinearSegmentedColormap.from_list("terr", [
    (0.00, "#6f9b55"), (0.12, "#a9b96c"), (0.30, "#dccb92"),
    (0.55, "#b98a56"), (0.78, "#8c6748"), (1.00, "#f4f1ec")])
SEA = np.array([0.66, 0.80, 0.91])


def clearance(ft):
    """Jeppesen MORA rule applied to a terrain height in ft."""
    add = 1000 if ft <= 5000 else 2000
    return int(math.ceil((ft + add) / 100.0) * 100)


def msa_value(ft):
    return int(math.ceil((ft + 1000) / 100.0) * 100)


# ---------------------------------------------------------------- geodesy
def gc_dist(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def dest(lat, lon, brg, dkm):
    p1, l1, b, d = math.radians(lat), math.radians(lon), math.radians(brg), dkm / R_EARTH
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), (math.degrees(l2) + 540) % 360 - 180


def gc_interp(lat1, lon1, lat2, lon2, f):
    p1, l1, p2, l2 = map(math.radians, (lat1, lon1, lat2, lon2))
    d = gc_dist(lat1, lon1, lat2, lon2) / R_EARTH
    if d < 1e-12:
        return lat1, lon1
    a, b = math.sin((1 - f) * d) / math.sin(d), math.sin(f * d) / math.sin(d)
    x = a * math.cos(p1) * math.cos(l1) + b * math.cos(p2) * math.cos(l2)
    y = a * math.cos(p1) * math.sin(l1) + b * math.cos(p2) * math.sin(l2)
    z = a * math.sin(p1) + b * math.sin(p2)
    return math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x))


def fmt_coord(lat, lon):
    def dm(v, pos, neg, w):
        h = pos if v >= 0 else neg
        v = abs(v); d = int(v); m = (v - d) * 60
        return f"{h}{d:0{w}d}°{m:04.1f}'"
    return f"{dm(lat, 'N', 'S', 2)} {dm(lon, 'E', 'W', 3)}"


# ---------------------------------------------------------------- DEM
def tile_name(la, lo):
    return f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"


class DEM:
    def __init__(self, cache):
        self.cache = cache
        os.makedirs(cache, exist_ok=True)
        self.mem, self.maxes, self.spikes = {}, {}, {}

    def _path(self, la, lo):
        return os.path.join(self.cache, tile_name(la, lo) + ".npy")

    def fetch(self, la, lo):
        p = self._path(la, lo)
        if os.path.exists(p) or os.path.exists(p + ".none"):
            return
        n = tile_name(la, lo)
        try:
            data = urllib.request.urlopen(TILE_URL.format(d=n[:3], n=n), timeout=180).read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):          # no tile = open ocean
                open(p + ".none", "w").close()
                return
            raise
        raw = np.frombuffer(gzip.decompress(data), dtype=">i2")
        side = math.isqrt(raw.size)
        a = raw.reshape(side, side)[: side - 1, : side - 1].astype(np.int16)
        void = a == -32768
        if void.any():                         # conservative void fill
            a = a.copy(); a[void] = a[~void].max() if (~void).any() else 0
        k = (side - 1) // N
        a = a.reshape(N, k, N, k).max(axis=(1, 3))   # max-preserving downsample
        np.save(p, a)

    def prefetch(self, cells, workers=8):
        todo = [c for c in cells if not (os.path.exists(self._path(*c)) or os.path.exists(self._path(*c) + ".none"))]
        if todo:
            print(f"  downloading {len(todo)} DEM tiles ...", flush=True)
        with cf.ThreadPoolExecutor(workers) as ex:
            list(ex.map(lambda c: self.fetch(*c), todo))

    def get(self, la, lo):
        key = (la, lo)
        if key not in self.mem:
            self.fetch(la, lo)
            p = self._path(la, lo)
            if os.path.exists(p):
                arr, nfix = despike(np.load(p))
                self.spikes[key] = nfix
                self.mem[key] = arr
            else:
                self.mem[key] = None
        return self.mem[key]

    def cell_max_m(self, la, lo):
        if (la, lo) not in self.maxes:
            t = self.get(la, lo)
            self.maxes[(la, lo)] = int(t.max()) if t is not None else 0
        return self.maxes[(la, lo)]

    def point_m(self, lat, lon):
        la, lo = math.floor(lat), math.floor(lon)
        t = self.get(la, lo)
        if t is None:
            return 0
        iy = min(int((lat - la) * N), N - 1); jx = min(int((lon - lo) * N), N - 1)
        return int(t[N - 1 - iy, jx])

    def mosaic(self, lat0, lat1, lon0, lon1):
        """South-up int16 array (metres) covering the box, plus pixel-centre lats/lons."""
        iy0, iy1 = math.floor(lat0 * N), math.ceil(lat1 * N)
        jx0, jx1 = math.floor(lon0 * N), math.ceil(lon1 * N)
        out = np.zeros((iy1 - iy0, jx1 - jx0), np.int16)
        for tl in range(math.floor(iy0 / N), math.floor((iy1 - 1) / N) + 1):
            for tn in range(math.floor(jx0 / N), math.floor((jx1 - 1) / N) + 1):
                t = self.get(tl, tn)
                if t is None:
                    continue
                ts = t[::-1]
                a0, a1 = max(iy0, tl * N), min(iy1, (tl + 1) * N)
                b0, b1 = max(jx0, tn * N), min(jx1, (tn + 1) * N)
                out[a0 - iy0:a1 - iy0, b0 - jx0:b1 - jx0] = ts[a0 - tl * N:a1 - tl * N, b0 - tn * N:b1 - tn * N]
        return out, (np.arange(iy0, iy1) + 0.5) / N, (np.arange(jx0, jx1) + 0.5) / N


SPIKE_M = 300     # a 90 m pixel higher than ALL 8 neighbours by this much is an artefact
CLUSTER_M = 600   # a small blob higher than ALL pixels 2 cells (~180 m) away by this much is an artefact


def _ring_max(a, r):
    """Max over the square ring at Chebyshev distance r around each pixel."""
    P = np.pad(a, r, mode="edge"); h, w = a.shape; out = None
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if max(abs(dy), abs(dx)) != r:
                continue
            v = P[r + dy:r + dy + h, r + dx:r + dx + w]
            out = v.copy() if out is None else np.maximum(out, v)
    return out


def despike(a):
    """Remove SRTM void-fill spikes (replaced, conservatively, by the surrounding max).
    Returns (cleaned array, pixels replaced)."""
    a = a.astype(np.int32)
    fixed = 0
    for r, thr in ((1, SPIKE_M), (2, CLUSTER_M), (1, SPIKE_M)):
        for _ in range(3):
            rm = _ring_max(a, r)
            bad = a - rm > thr
            if not bad.any():
                break
            fixed += int(bad.sum())
            a = np.where(bad, rm, a)
    return a.astype(np.int16), fixed


def block_max(Z, f):
    if f <= 1:
        return Z
    ny, nx = (Z.shape[0] // f) * f, (Z.shape[1] // f) * f
    return Z[:ny, :nx].reshape(ny // f, f, nx // f, f).max(axis=(1, 3))


# ---------------------------------------------------------------- inputs
def load_airports(path):
    db = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rec = dict(name=r["name"], lat=float(r["latitude_deg"]), lon=float(r["longitude_deg"]),
                           elev=float(r["elevation_ft"]) if r.get("elevation_ft") else None)
            except ValueError:
                continue
            for k in ("ident", "icao_code", "gps_code"):
                c = (r.get(k) or "").strip().upper()
                if len(c) == 4 and (k == "ident" or c not in db):
                    db[c] = dict(rec, icao=c)
    return db


def parse_routes(path, apts):
    routes = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            wps = []
            for tok in r["waypoints"].split():
                if "/" in tok:
                    n, la, lo = tok.split("/")
                    wps.append(dict(name=n, lat=float(la), lon=float(lo), apt=None))
                else:
                    a = apts.get(tok.upper())
                    if not a:
                        raise SystemExit(f"Unknown airport/fix '{tok}' in route {r['route_id']}")
                    wps.append(dict(name=tok.upper(), lat=a["lat"], lon=a["lon"], apt=a))
            routes.append(dict(id=r["route_id"], desc=r.get("description", ""), wps=wps))
    return routes


# ---------------------------------------------------------------- analysis
def analyse_route(dem, rt):
    half = CORRIDOR_NM * NM
    wps = rt["wps"]
    pts, s, legidx = [], [], []
    legs, s0 = [], 0.0
    for i in range(len(wps) - 1):
        a, b = wps[i], wps[i + 1]
        d = gc_dist(a["lat"], a["lon"], b["lat"], b["lon"])
        n = max(1, math.ceil(d / 1.0))
        for k in range(0 if i == 0 else 1, n + 1):
            pts.append(gc_interp(a["lat"], a["lon"], b["lat"], b["lon"], k / n)); s.append(s0 + d * k / n); legidx.append(i)
        legs.append(dict(frm=a["name"], to=b["name"], dist_km=d, s0=s0,
                         trk=bearing(a["lat"], a["lon"], b["lat"], b["lon"]), peak=None))
        s0 += d
    total = s0
    prof_max = np.zeros(int(math.ceil(total)) + 2, np.int32)
    candidates = []
    # straight sub-segments of <= 20 km inside each leg
    for li, leg in enumerate(legs):
        a, b = wps[li], wps[li + 1]
        n = max(1, math.ceil(leg["dist_km"] / 20.0))
        for k in range(n):
            pa = gc_interp(a["lat"], a["lon"], b["lat"], b["lon"], k / n)
            pb = gc_interp(a["lat"], a["lon"], b["lat"], b["lon"], (k + 1) / n)
            sa = leg["s0"] + leg["dist_km"] * k / n
            lat0, lon0 = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
            kx = 111.32 * math.cos(math.radians(lat0)); ky = 110.574
            dla, dlo = half / ky + 0.01, half / kx + 0.01
            Z, la, lo = dem.mosaic(min(pa[0], pb[0]) - dla, max(pa[0], pb[0]) + dla,
                                   min(pa[1], pb[1]) - dlo, max(pa[1], pb[1]) + dlo)
            X, Y = np.meshgrid((lo - lon0) * kx, (la - lat0) * ky)
            ax_, ay_ = (pa[1] - lon0) * kx, (pa[0] - lat0) * ky
            vx, vy = (pb[1] - lon0) * kx - ax_, (pb[0] - lat0) * ky - ay_
            L2 = vx * vx + vy * vy; L = math.sqrt(L2)
            t = np.clip(((X - ax_) * vx + (Y - ay_) * vy) / L2, 0, 1)
            d = np.hypot(X - (ax_ + t * vx), Y - (ay_ + t * vy))
            m = d <= half
            zm = np.where(m, Z, np.int16(-32768))
            iy, ix = np.unravel_index(np.argmax(zm), Z.shape)
            cand = dict(m=int(Z[iy, ix]), lat=float(la[iy]), lon=float(lo[ix]),
                        off_nm=float(d[iy, ix] / NM), s=float(sa + t[iy, ix] * L), leg=li)
            candidates.append(cand)
            if leg["peak"] is None or cand["m"] > leg["peak"]["m"]:
                leg["peak"] = cand
            bins = np.clip(np.floor(sa + t[m] * L).astype(int), 0, len(prof_max) - 1)
            np.maximum.at(prof_max, bins, Z[m].astype(np.int32))
    for leg in legs:
        leg["peak_ft"] = leg["peak"]["m"] * M2FT
        leg["route_mora"] = clearance(leg["peak_ft"])
    # distinct peaks (>= 15 km apart)
    peaks = []
    for c in sorted(candidates, key=lambda c: -c["m"]):
        if all(gc_dist(c["lat"], c["lon"], p["lat"], p["lon"]) >= 15 for p in peaks):
            peaks.append(c)
        if len(peaks) >= 8:
            break
    # grid MORA cells crossed by the centreline
    cells = []
    for (lat, lon), si in zip(pts, s):
        c = (math.floor(lat), math.floor(lon))
        if not cells or cells[-1]["cell"] != c:
            if any(x["cell"] == c for x in cells):
                continue
            mft = dem.cell_max_m(*c) * M2FT
            cells.append(dict(cell=c, s=si, max_ft=mft, gmora=clearance(mft)))
    centre = np.array([dem.point_m(la_, lo_) for la_, lo_ in pts]) * M2FT
    gm_along = np.array([clearance(dem.cell_max_m(math.floor(la_), math.floor(lo_)) * M2FT) for la_, lo_ in pts])
    top = max(candidates, key=lambda c: c["m"])
    return dict(legs=legs, peaks=peaks, cells=cells, total_km=total, pts=pts, s=np.array(s),
                legidx=legidx, prof_max_ft=prof_max * M2FT, centre_ft=centre, gm_along=gm_along,
                top=top, top_ft=top["m"] * M2FT, route_mora=clearance(top["m"] * M2FT),
                max_gmora=max(c["gmora"] for c in cells))


def analyse_msa(dem, apt):
    r = MSA_NM * NM
    kx = 111.32 * math.cos(math.radians(apt["lat"]))
    Z, la, lo = dem.mosaic(apt["lat"] - r / 110.574 - 0.01, apt["lat"] + r / 110.574 + 0.01,
                           apt["lon"] - r / kx - 0.01, apt["lon"] + r / kx + 0.01)
    LO, LA = np.meshgrid(lo, la)
    p1, p2 = math.radians(apt["lat"]), np.radians(LA)
    dl = np.radians(LO - apt["lon"])
    a = np.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    dist = 2 * R_EARTH * np.arcsin(np.sqrt(a))
    brg = (np.degrees(np.arctan2(np.sin(dl) * np.cos(p2),
                                 math.cos(p1) * np.sin(p2) - math.sin(p1) * np.cos(p2) * np.cos(dl))) + 360) % 360
    inside = dist <= r
    res = dict(apt=apt, quads=[])
    zm = np.where(inside, Z, np.int16(-32768)); iy, ix = np.unravel_index(np.argmax(zm), Z.shape)
    res["max_ft"] = Z[iy, ix] * M2FT; res["max_pos"] = (float(la[iy]), float(lo[ix]))
    res["max_dist_nm"] = float(dist[iy, ix] / NM); res["msa"] = msa_value(res["max_ft"])
    for q0, lab in ((0, "NE 000-090"), (90, "SE 090-180"), (180, "SW 180-270"), (270, "NW 270-360")):
        mq = inside & (brg >= q0) & (brg < q0 + 90)
        zq = np.where(mq, Z, np.int16(-32768)); jy, jx = np.unravel_index(np.argmax(zq), Z.shape)
        ft = Z[jy, jx] * M2FT
        res["quads"].append(dict(label=lab, q0=q0, max_ft=ft, msa=msa_value(ft), pos=(float(la[jy]), float(lo[jx]))))
    return res


# ---------------------------------------------------------------- plotting
def shaded(Z, lats, lons, vmax_ft):
    zf = Z.astype(float) * M2FT
    dy = (lats[1] - lats[0]) * 110574 * M2FT
    dx = (lons[1] - lons[0]) * 111320 * math.cos(math.radians(lats.mean())) * M2FT
    rgb = LightSource(azdeg=315, altdeg=40).shade(zf, cmap=CMAP, vmin=0, vmax=vmax_ft, vert_exag=2.0,
                                                  dx=dx, dy=dy, blend_mode="soft")[..., :3]
    rgb[Z <= 0] = SEA
    return rgb, zf


def base_map(Z, lats, lons, title, figsize=(11, 9)):
    vmax = max(float(Z.max()) * M2FT, 3000.0)
    rgb, zf = shaded(Z, lats, lons, vmax)
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    h = (lats[1] - lats[0]) / 2
    ax.imshow(rgb, origin="lower", extent=[lons[0] - h, lons[-1] + h, lats[0] - h, lats[-1] + h], interpolation="bilinear")
    ax.set_aspect(1 / math.cos(math.radians(lats.mean())))
    ax.set_xlim(lons[0], lons[-1]); ax.set_ylim(lats[0], lats[-1])
    ax.set_title(title, fontsize=12, loc="left", fontweight="bold", pad=14)
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    sm = ScalarMappable(norm=Normalize(0, vmax), cmap=CMAP)
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02); cb.set_label("Terrain elevation (ft)")
    return fig, ax, zf


HALO = [pe.withStroke(linewidth=3, foreground="white")]


def draw_route(ax, A, lw=2.2):
    lat = [p[0] for p in A["pts"]]; lon = [p[1] for p in A["pts"]]
    ax.plot(lon, lat, color="#c2188f", lw=lw, zorder=5, label="Route centreline")
    half = CORRIDOR_NM * NM
    for side in (-90, 90):
        off = []
        for i, (la, lo) in enumerate(A["pts"]):
            j = min(i + 1, len(A["pts"]) - 1); k = max(j - 1, 0)
            b = bearing(*A["pts"][k], *A["pts"][j])
            off.append(dest(la, lo, b + side, half))
        ax.plot([o[1] for o in off], [o[0] for o in off], color="#c2188f", lw=0.9, ls="--", zorder=5,
                label="±10 NM corridor" if side == -90 else None)


def draw_circle(ax, lat, lon, rkm, **kw):
    c = [dest(lat, lon, b, rkm) for b in range(0, 361, 3)]
    ax.plot([p[1] for p in c], [p[0] for p in c], **kw)


def plot_overview(dem, rt, A, path):
    la = [p[0] for p in A["pts"]]; lo = [p[1] for p in A["pts"]]
    m = 0.55
    lat0, lat1, lon0, lon1 = min(la) - m, max(la) + m, min(lo) - m, max(lo) + m
    Z, lats, lons = dem.mosaic(lat0, lat1, lon0, lon1)
    f = max(1, math.ceil(max(Z.shape) / 1500))
    Z = block_max(Z, f); lats = lats[: Z.shape[0] * f].reshape(-1, f).mean(1); lons = lons[: Z.shape[1] * f].reshape(-1, f).mean(1)
    fig, ax, _ = base_map(Z, lats, lons, f"{rt['id']}  {rt['desc']}\nShaded relief, Grid MORA (hundreds of ft), ±10 NM corridor")
    for g in range(math.floor(lat0), math.ceil(lat1) + 1):
        ax.axhline(g, color="#333", lw=0.6, alpha=0.6)
    for g in range(math.floor(lon0), math.ceil(lon1) + 1):
        ax.axvline(g, color="#333", lw=0.6, alpha=0.6)
    for gy in range(math.floor(lat0), math.ceil(lat1)):
        for gx in range(math.floor(lon0), math.ceil(lon1)):
            cy, cx = gy + 0.5, gx + 0.5
            if lat0 < cy < lat1 and lon0 < cx < lon1:
                v = clearance(dem.cell_max_m(gy, gx) * M2FT)
                ax.text(cx, cy, f"{v // 100}", color="#7a1010", fontsize=22, fontweight="bold",
                        ha="center", va="center", alpha=0.85, path_effects=HALO, zorder=4)
    draw_route(ax, A)
    for w in rt["wps"]:
        ax.plot(w["lon"], w["lat"], "s" if w["apt"] else "^", color="#1b2a6b", ms=7, zorder=6)
        ax.text(w["lon"], w["lat"] + 0.06, w["name"], fontsize=9, fontweight="bold", color="#1b2a6b",
                ha="center", path_effects=HALO, zorder=7)
        if w["apt"]:
            draw_circle(ax, w["lat"], w["lon"], MSA_NM * NM, color="#1b2a6b", lw=0.9, ls=":", zorder=5)
    for i, p in enumerate(A["peaks"][:5]):
        ax.plot(p["lon"], p["lat"], "^", color="k", ms=8 if i == 0 else 6, zorder=7)
        ax.text(p["lon"] + 0.05, p["lat"], f"{p['m'] * M2FT:,.0f} ft", fontsize=8.5, fontweight="bold" if i == 0 else None,
                path_effects=HALO, zorder=7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=8, frameon=False)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_closeup(dem, rt, A, path, centre, title):
    lat_c, lon_c = centre
    hl = 0.27; hw = hl / math.cos(math.radians(lat_c))
    Z, lats, lons = dem.mosaic(lat_c - hl, lat_c + hl, lon_c - hw, lon_c + hw)
    fig, ax, zf = base_map(Z, lats, lons, title, figsize=(10, 9))
    top = zf.max()
    if top > 600:
        step = 500 if top < 6000 else 1000
        lv = np.arange(step, top + step, step)
        cs = ax.contour(lons, lats, zf, levels=lv, colors="#3b2a1a", linewidths=0.35, alpha=0.7)
        ax.clabel(cs, levels=lv[1::2] if len(lv) > 6 else lv, fontsize=6.5, fmt="%d")
    draw_route(ax, A, lw=2.6)
    for w in rt["wps"]:
        if abs(w["lat"] - lat_c) < hl and abs(w["lon"] - lon_c) < hw:
            ax.plot(w["lon"], w["lat"], "s", color="#1b2a6b", ms=8, zorder=6)
            ax.text(w["lon"], w["lat"] + 0.012, w["name"], fontsize=9, fontweight="bold", ha="center", path_effects=HALO)
    for p in A["peaks"]:
        if abs(p["lat"] - lat_c) < hl and abs(p["lon"] - lon_c) < hw:
            ax.plot(p["lon"], p["lat"], "^", color="k", ms=10, zorder=7)
            ax.text(p["lon"] + 0.008, p["lat"] + 0.008, f"{p['m'] * M2FT:,.0f} ft\n{fmt_coord(p['lat'], p['lon'])}\n{p['off_nm']:.1f} NM off track",
                    fontsize=8, fontweight="bold", path_effects=HALO, zorder=7)
    ax.set_xlim(lons[0], lons[-1]); ax.set_ylim(lats[0], lats[-1])
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_profile(rt, A, path):
    x = np.arange(len(A["prof_max_ft"])) / NM
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=130)
    ax.fill_between(x, 0, A["prof_max_ft"], color="#b98a56", alpha=0.55, step="mid", label="Highest terrain within ±10 NM")
    ax.plot(A["s"] / NM, A["centre_ft"], color="#5a3d22", lw=1, label="Terrain under centreline")
    ax.step(A["s"] / NM, A["gm_along"], where="post", color="#7a1010", lw=1.4, ls="--", label="Grid MORA (est.)")
    for leg in A["legs"]:
        x0, x1 = leg["s0"] / NM, (leg["s0"] + leg["dist_km"]) / NM
        ax.hlines(leg["route_mora"], x0, x1, color="#c2188f", lw=2.2, label="Route MORA (est.)" if leg is A["legs"][0] else None)
        ax.text((x0 + x1) / 2, leg["route_mora"] + 150, f"{leg['route_mora']:,} ft", color="#c2188f", ha="center", fontsize=8, fontweight="bold")
        ax.axvline(x0, color="#999", lw=0.6)
    for w, xx in zip(rt["wps"], [l["s0"] / NM for l in A["legs"]] + [A["total_km"] / NM]):
        ax.text(xx, -0.07, w["name"], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, fontweight="bold")
    p = A["top"]
    ax.annotate(f"{p['m'] * M2FT:,.0f} ft", (p["s"] / NM, p["m"] * M2FT), xytext=(0, 14), textcoords="offset points",
                ha="center", fontsize=8, fontweight="bold", arrowprops=dict(arrowstyle="-", lw=0.6))
    ax.set_xlim(0, A["total_km"] / NM); ax.set_ylim(0, max(A["gm_along"].max(), A["route_mora"]) * 1.3)
    ax.set_ylabel("ft AMSL"); ax.set_title(f"{rt['id']}  vertical terrain profile", loc="left", fontweight="bold")
    ax.set_xlabel("\nDistance along route (NM)"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left", ncol=4, framealpha=0.9)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_msa(dem, M, path):
    apt = M["apt"]; r = MSA_NM * NM
    hl = r / 110.574 * 1.12; hw = hl / math.cos(math.radians(apt["lat"]))
    Z, lats, lons = dem.mosaic(apt["lat"] - hl, apt["lat"] + hl, apt["lon"] - hw, apt["lon"] + hw)
    f = max(1, math.ceil(max(Z.shape) / 1100))
    Z = block_max(Z, f); lats = lats[: Z.shape[0] * f].reshape(-1, f).mean(1); lons = lons[: Z.shape[1] * f].reshape(-1, f).mean(1)
    fig, ax, _ = base_map(Z, lats, lons, f"{apt['icao']}  {apt['name']}\nMSA estimate, 25 NM radius, true quadrants", figsize=(9, 8.5))
    draw_circle(ax, apt["lat"], apt["lon"], r, color="#1b2a6b", lw=1.6)
    for b in (0, 90, 180, 270):
        e = dest(apt["lat"], apt["lon"], b, r)
        ax.plot([apt["lon"], e[1]], [apt["lat"], e[0]], color="#1b2a6b", lw=0.8, ls="--")
    for q in M["quads"]:
        c = dest(apt["lat"], apt["lon"], q["q0"] + 45, r * 0.6)
        ax.text(c[1], c[0], f"{q['msa']:,}", fontsize=17, fontweight="bold", color="#1b2a6b", ha="center", va="center", path_effects=HALO)
        ax.plot(q["pos"][1], q["pos"][0], "^", color="k", ms=6)
        ax.text(q["pos"][1], q["pos"][0] - 0.03, f"{q['max_ft']:,.0f}", fontsize=7.5, ha="center", va="top", path_effects=HALO)
    ax.plot(apt["lon"], apt["lat"], "s", color="#c2188f", ms=8)
    ax.text(0.98, 0.02, f"Omnidirectional MSA {M['msa']:,} ft", transform=ax.transAxes, ha="right", fontsize=10,
            fontweight="bold", path_effects=HALO)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- report
CSS = """
/* Layout: one reading column, chart-room palette: Jeppesen maroon for MORA, route magenta, navy for fixes */
:root{--bg:#f4f5f2;--fg:#1c1f24;--mut:#5b6169;--card:#ffffff;--line:#d8dbd4;--mora:#7a1414;--route:#b0157f;--fix:#1b2a6b;
--warn:#fff3d1;--warnb:#c99300;--th:#eceee8;--display:"Barlow Semi Condensed","Arial Narrow",sans-serif;
--body:"Barlow",system-ui,"Segoe UI",sans-serif;--ar:"Noto Sans Arabic","Segoe UI",Tahoma,sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15171b;--fg:#e8e9ec;--mut:#a0a6ae;--card:#1e2126;--line:#353a41;
--mora:#ff8f86;--route:#f06ac4;--fix:#9fb2ff;--warn:#3a3218;--warnb:#d6a91c;--th:#262a30;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#15171b;--fg:#e8e9ec;--mut:#a0a6ae;--card:#1e2126;--line:#353a41;
--mora:#ff8f86;--route:#f06ac4;--fix:#9fb2ff;--warn:#3a3218;--warnb:#d6a91c;--th:#262a30;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 var(--body)}
main{max-width:1120px;margin:0 auto;padding-block:28px 80px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:6px}
main>*{min-width:0}
h1,h2,h3{font-family:var(--display);text-wrap:balance;line-height:1.15;margin:0}
h1{font-size:clamp(28px,5vw,40px);font-weight:700}
h2{font-size:26px;font-weight:700;margin-top:40px;padding-bottom:6px;border-bottom:2px solid var(--line)}
h3{font-size:18px;font-weight:600;margin-top:22px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut)}
p{margin:6px 0;max-width:70ch}.mut{color:var(--mut)}
.eyebrow{font:600 13px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--route)}
.warn{background:var(--warn);border-left:4px solid var(--warnb);padding:12px 16px;border-radius:4px;margin:8px 0}
.ar{direction:rtl;text-align:right;font-family:var(--ar)}
.wrap{overflow-x:auto;margin:6px 0}
table{border-collapse:collapse;width:100%;background:var(--card);font-size:14.5px}
th,td{border:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}
th{background:var(--th);font-family:var(--display);font-weight:600;letter-spacing:.02em}
td.n{text-align:right;font:14px var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
.big{font-weight:600;color:var(--mora)}
a{color:var(--fix)}a:focus-visible{outline:2px solid var(--route);outline-offset:2px}
figure{margin:10px 0}img{display:block;max-width:100%;height:auto;border:1px solid var(--line);border-radius:4px;background:#fff}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:14px}
.key{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:14px;color:var(--mut)}
.key b{font-family:var(--mono);color:var(--fg)}
"""
FONTS = ("<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=Barlow:wght@400;600&"
         "family=Barlow+Semi+Condensed:wght@600;700&family=IBM+Plex+Mono:wght@400;600&family=Noto+Sans+Arabic:wght@400;700&display=swap'>")


def e(x):
    return html.escape(str(x))


def write_report(out, routes, results, msas, dem):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    H = ["<title>Red Sea Terrain Clearance</title><meta name='claude:viewport-fit-sides' content='cover'>",
         FONTS, f"<style>{CSS}</style><main>"]
    H.append("<p class='eyebrow'>Terrain clearance estimates</p><h1>Red Sea Terrain Clearance</h1>")
    H.append(f"<p class='mut'>Highest terrain, Grid MORA, Route MORA and MSA for {len(routes)} routes and {len(msas)} airports. Generated {now}.</p>")
    demo = any("DEMO" in rt["desc"] for rt in routes)
    if demo:
        H.append("<div class='warn ar'><b>رحلات تجريبية:</b> المسارات هنا خطوط مباشرة (great circle) بين المطارات، "
                 "مش الـ airways الفعلية. ابعت قائمة رحلات الشركة بنقاط المسار (waypoints) علشان التقرير يتعمل على المسارات الحقيقية.</div>")
    H.append("<div class='warn ar'><b>تنبيه مهم:</b> القيم دي <b>تقديرية</b> محسوبة من نموذج تضاريس SRTM المفتوح فقط "
             "(دقة رأسية تقريبًا ±16 م)، ومش شاملة العوائق الصناعية زي الأبراج والهوائيات. "
             "مش بديل عن قيم Jeppesen أو الـ AIP المنشورة، وتُستخدم للمراجعة والتخطيط فقط.</div>")
    H.append("<div class='warn'><b>Not for operational use.</b> Terrain-only estimates from the SRTM 1&quot; DEM, "
             "max-preserved at 3&quot; (~90 m) resolution. Man-made obstacles are not included. Always use the published "
             "Jeppesen / AIP Grid MORA, Route MORA and MSA. Rules applied: MORA = highest terrain + 1,000 ft "
             "(terrain ≤ 5,000 ft) or + 2,000 ft (terrain &gt; 5,000 ft), rounded up to 100 ft; Route MORA uses a ±10 NM corridor; "
             "MSA = highest terrain within 25 NM of the airport reference point + 1,000 ft, rounded up to 100 ft, in true-bearing quadrants "
             "(published MSA sectors are magnetic and may be centred on a navaid).</div>")
    nfix = sum(dem.spikes.values())
    H.append(f"<p class='mut'>DEM quality control: {nfix:,} spike artefact pixels removed from {len(dem.spikes)} tiles "
             f"(pixel higher than every neighbour by &gt;{SPIKE_M} m, or a small blob &gt;{CLUSTER_M} m above its surroundings).</p>")
    H.append("<h2>Summary</h2><div class='wrap'><table><tr><th>Route</th><th>Description</th><th>Distance</th>"
             "<th>Highest terrain ±10 NM</th><th>Position</th><th>Route MORA est.</th><th>Max Grid MORA est.</th></tr>")
    for rt, A in zip(routes, results):
        t = A["top"]
        H.append(f"<tr><td><a href='#{e(rt['id'])}'><b>{e(rt['id'])}</b></a></td><td>{e(rt['desc'])}</td>"
                 f"<td class='n'>{A['total_km'] / NM:,.0f} NM</td><td class='n'>{A['top_ft']:,.0f} ft</td>"
                 f"<td>{fmt_coord(t['lat'], t['lon'])}<br><span class='mut'>{t['off_nm']:.1f} NM off track</span></td>"
                 f"<td class='n big'>{A['route_mora']:,} ft</td><td class='n big'>{A['max_gmora']:,} ft</td></tr>")
    H.append("</table></div>")
    top = max(zip(routes, results), key=lambda x: x[1]["top_ft"])
    H.append("<div class='ar'><h3 style='text-transform:none;letter-spacing:0'>الخلاصة</h3>"
             f"<p>أعلى تضاريس على كل المسارات {top[1]['top_ft']:,.0f} قدم على مسار {e(top[0]['id'])}، "
             f"و Route MORA التقديري ليه {top[1]['route_mora']:,} قدم. "
             "الأرقام الكبيرة باللون الأحمر على الخرايط هي Grid MORA بالمئات (مثلاً 107 = 10,700 قدم). "
             "الخط البنفسجي هو المسار والخطوط المتقطعة حدود ±10 NM.</p></div>")
    H.append("<h2>Airport MSA estimates (25 NM)</h2><div class='wrap'><table><tr><th>Airport</th><th>Elev.</th>"
             "<th>Highest terrain ≤25 NM</th><th>MSA omni</th><th>NE</th><th>SE</th><th>SW</th><th>NW</th></tr>")
    for M in msas:
        a = M["apt"]
        H.append(f"<tr><td><a href='#msa-{e(a['icao'])}'><b>{e(a['icao'])}</b></a> {e(a['name'])}</td>"
                 f"<td class='n'>{a['elev'] or 0:,.0f} ft</td><td class='n'>{M['max_ft']:,.0f} ft<br>"
                 f"<span class='mut'>{M['max_dist_nm']:.1f} NM from ARP</span></td><td class='n big'>{M['msa']:,} ft</td>"
                 + "".join(f"<td class='n'>{q['msa']:,}</td>" for q in M["quads"]) + "</tr>")
    H.append("</table></div>")
    for rt, A in zip(routes, results):
        rid = rt["id"]
        H.append(f"<h2 id='{e(rid)}'>{e(rid)} · {e(rt['desc'])}</h2>")
        H.append(f"<p class='mut'>Waypoints: {e(' → '.join(w['name'] for w in rt['wps']))} · {A['total_km'] / NM:,.0f} NM</p>")
        H.append("<h3>Legs</h3><div class='wrap'><table><tr><th>Leg</th><th>True trk</th><th>Dist</th><th>Highest terrain ±10 NM</th>"
                 "<th>Position</th><th>Route MORA est.</th></tr>")
        for L in A["legs"]:
            p = L["peak"]
            H.append(f"<tr><td>{e(L['frm'])} → {e(L['to'])}</td><td class='n'>{L['trk']:03.0f}°</td><td class='n'>{L['dist_km'] / NM:,.0f} NM</td>"
                     f"<td class='n'>{L['peak_ft']:,.0f} ft</td><td>{fmt_coord(p['lat'], p['lon'])} · {p['off_nm']:.1f} NM off · "
                     f"{p['s'] / NM:,.0f} NM along</td><td class='n big'>{L['route_mora']:,} ft</td></tr>")
        H.append("</table></div><h3>Grid MORA cells crossed</h3><div class='wrap'><table><tr><th>Cell (SW corner)</th>"
                 "<th>Entered at</th><th>Highest terrain in cell</th><th>Grid MORA est.</th></tr>")
        for c in A["cells"]:
            la, lo = c["cell"]
            H.append(f"<tr><td>{'N' if la >= 0 else 'S'}{abs(la):02d} {'E' if lo >= 0 else 'W'}{abs(lo):03d}</td>"
                     f"<td class='n'>{c['s'] / NM:,.0f} NM</td><td class='n'>{c['max_ft']:,.0f} ft</td>"
                     f"<td class='n big'>{c['gmora']:,} ft <span class='mut'>({c['gmora'] // 100})</span></td></tr>")
        H.append("</table></div><h3>Highest terrain points in corridor</h3><div class='wrap'><table><tr><th>#</th><th>Elevation</th>"
                 "<th>Position</th><th>Off track</th><th>Along track</th></tr>")
        for i, p in enumerate(A["peaks"], 1):
            H.append(f"<tr><td>{i}</td><td class='n'>{p['m'] * M2FT:,.0f} ft <span class='mut'>({p['m']:,} m)</span></td>"
                     f"<td>{fmt_coord(p['lat'], p['lon'])}</td><td class='n'>{p['off_nm']:.1f} NM</td><td class='n'>{p['s'] / NM:,.0f} NM</td></tr>")
        H.append("</table></div>")
        H.append("<div class='key'><span><b>107</b> Grid MORA in hundreds of ft</span><span>magenta line: route</span>"
                 "<span>dashed: ±10 NM corridor</span><span>dotted circle: 25 NM MSA area</span><span>▲ highest terrain</span></div>")
        H.append(f"<img src='img/{e(rid)}_overview.png' alt='{e(rid)} overview map'>")
        H.append(f"<img src='img/{e(rid)}_profile.png' alt='{e(rid)} vertical profile'>")
        H.append(f"<h3>Close-ups</h3><div class='grid2'>" + "".join(
            f"<img src='img/{e(rid)}_closeup{k}.png' alt='{e(rid)} close-up {k}'>" for k in range(1, A["n_closeups"] + 1)) + "</div>")
    H.append("<h2>Airport MSA diagrams</h2><div class='grid2'>")
    for M in msas:
        H.append(f"<div id='msa-{e(M['apt']['icao'])}'><img src='img/MSA_{e(M['apt']['icao'])}.png' alt='MSA {e(M['apt']['icao'])}'></div>")
    H.append("</div><p class='mut'>Data: SRTM GL1 via AWS Terrain Tiles (skadi); airports: OurAirports. "
             "Generated by tools/terrain_report.py.</p></main>")
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(H))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--routes", default="routes.csv")
    ap.add_argument("--out", default="output/terrain_report")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/terrain_report"))
    ap.add_argument("--airports", default=None, help="OurAirports airports.csv (downloaded to cache if omitted)")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "img"), exist_ok=True)
    apath = args.airports or os.path.join(args.cache, "airports.csv")
    if not os.path.exists(apath):
        os.makedirs(args.cache, exist_ok=True)
        print("downloading airport database ...")
        urllib.request.urlretrieve(AIRPORTS_URL, apath)
    apts = load_airports(apath)
    routes = parse_routes(args.routes, apts)
    dem = DEM(os.path.join(args.cache, "dem"))
    # prefetch every tile that any map/analysis will touch
    cells = set()
    for rt in routes:
        la = [w["lat"] for w in rt["wps"]]; lo = [w["lon"] for w in rt["wps"]]
        for y in range(math.floor(min(la) - 0.8), math.floor(max(la) + 0.8) + 1):
            for x in range(math.floor(min(lo) - 0.9), math.floor(max(lo) + 0.9) + 1):
                cells.add((y, x))
    dem.prefetch(sorted(cells))
    results = []
    for rt in routes:
        print(f"route {rt['id']} ...", flush=True)
        A = analyse_route(dem, rt)
        img = os.path.join(args.out, "img")
        plot_overview(dem, rt, A, os.path.join(img, f"{rt['id']}_overview.png"))
        plot_profile(rt, A, os.path.join(img, f"{rt['id']}_profile.png"))
        # close-up 1: highest terrain; close-up 2: highest terrain on a different leg / >60 km away
        spots = [A["top"]] + [p for p in A["peaks"] if gc_dist(p["lat"], p["lon"], A["top"]["lat"], A["top"]["lon"]) > 60][:1]
        for k, p in enumerate(spots, 1):
            plot_closeup(dem, rt, A, os.path.join(img, f"{rt['id']}_closeup{k}.png"), (p["lat"], p["lon"]),
                         f"{rt['id']}  close-up {k}: {p['m'] * M2FT:,.0f} ft terrain, {p['off_nm']:.1f} NM off track")
        A["n_closeups"] = len(spots)
        results.append(A)
    msas, seen = [], set()
    for rt in routes:
        for w in rt["wps"]:
            if w["apt"] and w["name"] not in seen:
                seen.add(w["name"]); print(f"MSA {w['name']} ...", flush=True)
                M = analyse_msa(dem, w["apt"]); plot_msa(dem, M, os.path.join(args.out, "img", f"MSA_{w['name']}.png")); msas.append(M)
    write_report(args.out, routes, results, msas, dem)
    summary = [dict(route=rt["id"], distance_nm=round(A["total_km"] / NM, 1), highest_terrain_ft=round(A["top_ft"]),
                    highest_lat=A["top"]["lat"], highest_lon=A["top"]["lon"], route_mora_ft=A["route_mora"],
                    max_grid_mora_ft=A["max_gmora"],
                    legs=[dict(frm=L["frm"], to=L["to"], highest_ft=round(L["peak_ft"]), route_mora_ft=L["route_mora"]) for L in A["legs"]])
               for rt, A in zip(routes, results)]
    summary_msa = [dict(airport=M["apt"]["icao"], highest_ft=round(M["max_ft"]), msa_ft=M["msa"],
                        quadrants={q["label"]: q["msa"] for q in M["quads"]}) for M in msas]
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(dict(routes=summary, msa=summary_msa), f, indent=2)
    print("report:", os.path.join(args.out, "index.html"))


if __name__ == "__main__":
    main()
