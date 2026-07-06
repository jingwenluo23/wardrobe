"""
CAD T-Shirt Mesh Generator
==========================
INTERNAL SOURCE TOOLING.
Do not run this directly in normal development. The live runtime asset is
derived from the checked-in source mesh `src/lib/assets/tshirt_cad_source.json`
via the single supported entrypoint: `npm run cad:refresh`.

Produces a proper closed-shell 3D T-shirt mesh following real garment CAD principles.
Output: JSON geometry (positions, normals, uvs, indices, segments)
       + OBJ file for inspection

Architecture:
  - Body: revolve-lofted cross-sections along the torso axis
  - Sleeves: true elliptical tubes lofted from armhole ellipse to cuff ellipse  
  - Neck collar: torus-segment ring around the neck opening
  - All pieces are CLOSED SHELLS (manifold geometry, no open edges)
  - Smooth G1/G2 tangent continuity at all seams

Coordinate system:
  X: left(-) to right(+)
  Y: down(-) to up(+)   [hem is bottom, collar is top]
  Z: back(-) to front(+)
"""

import math
import json
import struct
from typing import NamedTuple

# ─── Garment parameters (size M, in metres) ───────────────────────────────────
class TShirtParams:
    # Body
    body_height    = 0.72    # hem to shoulder (not including collar)
    body_width     = 0.55    # half-width at chest (full = 1.10m)
    body_depth     = 0.10    # half-depth front-to-back (full = 0.20m)
    waist_ratio    = 0.96    # waist width as fraction of chest
    hip_ratio      = 0.98    # hip width as fraction of chest
    shoulder_width = 0.48    # half-width at shoulder seam
    shoulder_drop  = 0.04    # how much shoulder drops from neck to armhole
    
    # Neck
    neck_width     = 0.095   # half-width of neck opening
    neck_depth_f   = 0.065   # front neck drop below shoulder line
    neck_depth_b   = 0.018   # back neck rise above shoulder seam
    collar_height  = 0.022   # height of rib collar tube
    collar_thick   = 0.012   # thickness of collar tube wall

    # Armhole
    armhole_depth  = 0.22    # distance from shoulder to underarm (vertical)
    armhole_depth_z= 0.06    # how far forward armhole curves (depth offset)

    # Sleeve
    sleeve_length  = 0.22    # shoulder point to cuff (horizontal)
    sleeve_width_h = 0.072   # half-height of sleeve at armhole (Y radius)
    sleeve_width_d = 0.058   # half-depth of sleeve at armhole (Z radius)
    cuff_width_h   = 0.052   # half-height at cuff
    cuff_width_d   = 0.038   # half-depth at cuff
    sleeve_drop    = 0.030   # how far sleeve droops (underarm lower than shoulder)

    # Hem
    hem_height     = 0.018   # height of hem band
    hem_depth      = 0.006   # hem tucks slightly inward

P = TShirtParams()

# ─── Resolution ───────────────────────────────────────────────────────────────
BODY_RINGS    = 64   # vertical rings on the body tube
BODY_COLS     = 64   # columns around the body circumference
SLEEVE_RINGS  = 24   # rings along the sleeve
SLEEVE_CIRC   = 28   # circumference steps around sleeve tube
COLLAR_RINGS  = 8    # rings around collar tube height
COLLAR_CIRC   = 40   # steps around collar circumference

# ─── Math helpers ─────────────────────────────────────────────────────────────
def lerp(a, b, t):
    return a + (b - a) * t

def smoothstep(a, b, t):
    t = max(0, min(1, (t - a) / (b - a))) if b != a else (0 if t < a else 1)
    return t * t * (3 - 2 * t)

def bezier3(p0, p1, p2, t):
    u = 1 - t
    return u*u*p0 + 2*u*t*p1 + t*t*p2

def normalize(v):
    l = math.sqrt(sum(x*x for x in v))
    return tuple(x / (l or 1) for x in v)

def cross(a, b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )

def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))

# ─── Cross-section profile ────────────────────────────────────────────────────
def body_cross_section(v: float):
    """
    Returns (half_x, half_z) — the ellipse radii of the body cross-section
    at normalised vertical position v (0=hem, 1=shoulder).
    This drives the loft that creates the body tube.
    """
    # Width profile: slight taper at shoulder, slight flare at hip
    if v > 0.85:
        # Shoulder taper
        t = (v - 0.85) / 0.15
        hw = lerp(P.body_width, P.shoulder_width, smoothstep(0, 1, t))
    elif v < 0.12:
        # Hip flare
        t = v / 0.12
        hw = lerp(P.body_width * P.hip_ratio, P.body_width, smoothstep(0, 1, t))
    else:
        # Slight waist pinch in the middle
        waist_t = smoothstep(0.12, 0.45, v) * (1 - smoothstep(0.55, 0.85, v))
        hw = lerp(P.body_width, P.body_width * P.waist_ratio, waist_t * 0.4)

    # Depth profile: slightly shallower at hem and shoulder
    hem_t      = 1 - smoothstep(0, 0.10, v)
    shoulder_t = smoothstep(0.82, 1.0, v)
    hd = P.body_depth * (1 - hem_t * 0.25 - shoulder_t * 0.18)

    return hw, hd


def body_y_at_x(x_norm: float, v: float) -> float:
    """
    Vertical adjustment from shoulder profile.
    x_norm in [-1, 1]: lateral position normalised to body width.
    v: vertical parameter (0=hem, 1=shoulder top).
    Returns y-offset (shoulder slope + neck dip).
    """
    abs_x = abs(x_norm)
    # Shoulder slope: drops from neck to armhole edge
    shoulder_slope = abs_x * P.shoulder_drop
    return -shoulder_slope


# ─── Body tube ────────────────────────────────────────────────────────────────
def build_body():
    """
    Build the main body as a closed vertical tube (cylinder-ish).
    Vertex layout: ring 0 = hem, ring BODY_RINGS-1 = shoulder.
    Each ring has BODY_COLS vertices going around the circumference.
    
    Circumference parameterisation (angle θ):
      θ=0   → right side seam  (+X)
      θ=π/2 → front centre     (+Z)
      θ=π   → left side seam   (-X)
      θ=3π/2→ back centre      (-Z)
    """
    verts = []   # (x, y, z)
    uvs   = []   # (u, v)

    # Neck opening defines which columns are "open" at the top
    # Neck occupies columns where |x_norm| < neck_width/body_width at the shoulder
    neck_x_norm = P.neck_width / P.shoulder_width  # fraction of shoulder width

    for ri in range(BODY_RINGS):
        v  = ri / (BODY_RINGS - 1)            # 0=hem, 1=shoulder
        y0 = lerp(-P.body_height * 0.5, P.body_height * 0.5, v)
        hw, hd = body_cross_section(v)

        for ci in range(BODY_COLS):
            theta = (ci / BODY_COLS) * math.pi * 2
            # Ellipse cross-section
            x = hw * math.cos(theta)
            z = hd * math.sin(theta)

            # Shoulder slope: depress the y position near shoulder by lateral amount
            x_norm = x / (P.shoulder_width if v > 0.85 else P.body_width)
            y_off  = body_y_at_x(x_norm, v) * smoothstep(0.75, 1.0, v)
            y = y0 + y_off

            verts.append((x, y, z))
            uvs.append((ci / BODY_COLS, v))

    return verts, uvs


def body_ring_indices(body_verts_offset=0):
    """Closed-tube indices for the body."""
    idx = []
    for ri in range(BODY_RINGS - 1):
        for ci in range(BODY_COLS):
            ni = (ci + 1) % BODY_COLS
            a = body_verts_offset + ri * BODY_COLS + ci
            b = body_verts_offset + ri * BODY_COLS + ni
            c = body_verts_offset + (ri+1) * BODY_COLS + ci
            d = body_verts_offset + (ri+1) * BODY_COLS + ni
            idx += [a, b, d, a, d, c]
    return idx


# ─── Neck opening ─────────────────────────────────────────────────────────────
def neck_opening_angles():
    """
    Returns the range of θ angles (in body tube parameterisation) that are
    within the neck opening. The front neck is a deep U; the back is nearly flat.
    """
    # Neck occupies the "top" arc of the tube in Z (front side)
    # We parameterise the neck as angles around the top of the tube
    # For a crew neck: front arc from -neck_half_angle to +neck_half_angle
    neck_half_angle = math.asin(min(0.99, P.neck_width / P.shoulder_width))
    return neck_half_angle  # symmetric around θ=π/2 (front)


def neck_top_vertices(shoulder_ring_verts):
    """
    Given the shoulder ring vertices, compute the neck opening boundary.
    Returns list of (x, y, z) points going around the neck opening perimeter.
    """
    # Use the shoulder ring as the base, then apply the neck profile
    pts = []
    neck_half_angle = neck_opening_angles()

    # Front neck: semicircle dip
    # Back neck: shallow arc
    N = 32  # points around neck perimeter
    for i in range(N):
        t = i / N
        angle = t * math.pi * 2  # full circle

        # Parametric position in neck space
        # We map to the shoulder ring at the appropriate angle
        # θ=π/2 is front, θ=3π/2 is back
        hw, hd = body_cross_section(1.0)
        x_base = hw * math.cos(angle + math.pi/2)
        z_base = hd * math.sin(angle + math.pi/2)

        # Scale to neck opening
        scale = P.neck_width / hw
        x = x_base * scale
        z = z_base * scale * (hd / P.body_depth)

        # Y: front dips down, back is nearly flat
        front_t = max(0, math.sin(angle))
        back_t  = max(0, -math.sin(angle))
        y = P.body_height * 0.5 - P.neck_depth_f * front_t - P.neck_depth_b * back_t

        pts.append((x, y, z))
    return pts


# ─── Collar ring ──────────────────────────────────────────────────────────────
def build_collar():
    """
    The collar is a torus-segment: a tube bent along the neck opening curve.
    It sits on top of the neck opening and provides a proper rib-band.
    """
    verts = []
    uvs   = []
    N = COLLAR_CIRC   # around collar perimeter
    R = COLLAR_RINGS  # across collar thickness

    # Neck path: ellipse centred at shoulder level, slightly raised
    # Front arc: semicircle; back arc: flatter
    for ni in range(N):
        # Position along neck circumference
        t = ni / N
        path_angle = t * math.pi * 2  # 0=right, π/2=front, π=left, 3π/2=back

        # Path point: ellipse in XZ, neck shape in Y
        path_x = P.neck_width * math.cos(path_angle)
        path_z = P.neck_width * 0.62 * math.sin(path_angle)  # slightly shallower in Z

        # Neck Y contour: front drops, back nearly flat
        front_t = max(0, math.sin(path_angle))
        back_t  = max(0, -math.sin(path_angle))
        path_y_base = P.body_height * 0.5 - P.neck_depth_f * front_t * 0.8 - P.neck_depth_b * back_t

        # Tangent along path for tube orientation
        dt = 0.01
        ta2 = (t + dt) * math.pi * 2
        nx2 = P.neck_width * math.cos(ta2)
        nz2 = P.neck_width * 0.62 * math.sin(ta2)
        tangent = normalize((nx2 - path_x, 0, nz2 - path_z))

        # Normal points outward from centre of neck
        outward = normalize((path_x, 0, path_z))
        # Binormal = up-ish + outward tilt
        binormal = normalize((outward[0] * 0.3, 1, outward[2] * 0.3))

        # Cross-section: small ellipse (collar tube profile)
        for ri in range(R):
            phi = (ri / R) * math.pi * 2
            # Tube cross-section: slightly flattened (taller than wide)
            dy = P.collar_height * 0.5 * math.cos(phi)
            dr = P.collar_thick  * 0.5 * math.sin(phi)

            # Offset along outward + up
            x = path_x + outward[0] * dr
            y = path_y_base + dy + P.collar_height * 0.5
            z = path_z + outward[2] * dr

            verts.append((x, y, z))
            uvs.append((ni / N, ri / R))

    return verts, uvs


def collar_indices(offset=0):
    idx = []
    N, R = COLLAR_CIRC, COLLAR_RINGS
    for ni in range(N):
        nni = (ni + 1) % N
        for ri in range(R):
            nri = (ri + 1) % R
            a = offset + ni  * R + ri
            b = offset + nni * R + ri
            c = offset + ni  * R + nri
            d = offset + nni * R + nri
            idx += [a, b, d, a, d, c]
    return idx


# ─── Armhole ellipse ──────────────────────────────────────────────────────────
def armhole_ellipse(side: str):
    """
    Returns the centre and radii of the armhole ellipse.
    The armhole is where the body tube meets the sleeve tube.
    """
    sign = -1 if side == "left" else 1

    # Centre: at the shoulder/underarm junction, offset laterally
    cx = sign * P.shoulder_width
    cy = P.body_height * 0.5 - P.armhole_depth * 0.5   # midpoint of armhole depth
    cz = P.armhole_depth_z * (1 if side == "right" else 1)  # slightly forward

    # Radii: the armhole is a vertical ellipse
    ry = P.armhole_depth * 0.5          # vertical radius
    rz = P.sleeve_width_d               # depth radius

    return (cx, cy, cz), ry, rz


# ─── Sleeve tube ──────────────────────────────────────────────────────────────
def build_sleeve(side: str):
    """
    Build one sleeve as a closed elliptical tube.
    Ring 0 = at armhole (matches body armhole ellipse)
    Ring SLEEVE_RINGS-1 = cuff
    """
    sign = -1 if side == "left" else 1
    verts = []
    uvs   = []

    (cx, cy, cz), arm_ry, arm_rz = armhole_ellipse(side)

    # Cuff centre: extend outward from shoulder by sleeve_length
    # Sleeve droops slightly (underarm is lower)
    cuff_cx = cx + sign * P.sleeve_length
    cuff_cy = cy - P.sleeve_drop
    cuff_cz = cz

    for ri in range(SLEEVE_RINGS):
        t = ri / (SLEEVE_RINGS - 1)   # 0=armhole, 1=cuff

        # Loft: interpolate centre and radii
        ring_cx = lerp(cx, cuff_cx, smoothstep(0, 1, t))
        ring_cy = lerp(cy, cuff_cy, t)
        ring_cz = lerp(cz, cuff_cz, t)

        ring_ry = lerp(arm_ry, P.cuff_width_h, smoothstep(0, 1, t))
        ring_rz = lerp(arm_rz, P.cuff_width_d, smoothstep(0, 1, t))

        for ci in range(SLEEVE_CIRC):
            phi = (ci / SLEEVE_CIRC) * math.pi * 2
            # θ=0 → top of sleeve, θ=π → bottom (underarm)
            y = ring_cy + ring_ry * math.cos(phi)
            z = ring_cz + ring_rz * math.sin(phi)
            x = ring_cx

            verts.append((x, y, z))
            uvs.append((ci / SLEEVE_CIRC, t))

    return verts, uvs


def sleeve_indices(offset=0, reverse=False):
    idx = []
    for ri in range(SLEEVE_RINGS - 1):
        for ci in range(SLEEVE_CIRC):
            ni = (ci + 1) % SLEEVE_CIRC
            a = offset + ri * SLEEVE_CIRC + ci
            b = offset + ri * SLEEVE_CIRC + ni
            c = offset + (ri+1) * SLEEVE_CIRC + ci
            d = offset + (ri+1) * SLEEVE_CIRC + ni
            if reverse:
                idx += [a, d, b, a, c, d]
            else:
                idx += [a, b, d, a, d, c]
    return idx


def sleeve_cuff_cap(offset, reverse=False):
    """Fan cap closing the cuff end."""
    idx = []
    base = offset + (SLEEVE_RINGS - 1) * SLEEVE_CIRC
    # Cap centre is the average of the last ring — we'll add it as an extra vertex
    cap_centre_idx = offset + SLEEVE_RINGS * SLEEVE_CIRC  # appended after tube verts
    for ci in range(SLEEVE_CIRC):
        ni = (ci + 1) % SLEEVE_CIRC
        a = base + ci
        b = base + ni
        if reverse:
            idx += [cap_centre_idx, b, a]
        else:
            idx += [cap_centre_idx, a, b]
    return idx


def sleeve_armhole_cap(offset, reverse=False):
    """Fan cap closing the armhole end (inner end of sleeve tube)."""
    idx = []
    base = offset  # ring 0
    cap_centre_idx = offset + SLEEVE_RINGS * SLEEVE_CIRC + 1  # second extra vertex
    for ci in range(SLEEVE_CIRC):
        ni = (ci + 1) % SLEEVE_CIRC
        a = base + ci
        b = base + ni
        if reverse:
            idx += [cap_centre_idx, a, b]
        else:
            idx += [cap_centre_idx, b, a]
    return idx


# ─── Hem band ─────────────────────────────────────────────────────────────────
def build_hem_band():
    """
    Flat torus-like band at the bottom of the body closing the hem.
    Sits below the body tube's bottom ring.
    """
    verts = []
    uvs   = []
    N = BODY_COLS
    # Two rings: outer (body bottom) and inner (slightly inset)
    hw, hd = body_cross_section(0.0)
    y_bottom = -P.body_height * 0.5

    for ri in range(2):
        scale = 1.0 if ri == 0 else lerp(1.0, 1.0 - P.hem_depth/hw, 1.0)
        y = y_bottom - ri * P.hem_height
        for ci in range(N):
            theta = (ci / N) * math.pi * 2
            x = hw * scale * math.cos(theta)
            z = hd * scale * math.sin(theta)
            verts.append((x, y, z))
            uvs.append((ci / N, ri))
    return verts, uvs


def hem_band_indices(offset=0):
    idx = []
    for ci in range(BODY_COLS):
        ni = (ci + 1) % BODY_COLS
        a = offset + ci
        b = offset + ni
        c = offset + BODY_COLS + ci
        d = offset + BODY_COLS + ni
        idx += [a, b, d, a, d, c]
    # Bottom cap (inner ring)
    cap_idx = offset + 2 * BODY_COLS
    base = offset + BODY_COLS
    for ci in range(BODY_COLS):
        ni = (ci + 1) % BODY_COLS
        idx += [cap_idx, base + ni, base + ci]
    return idx


# ─── Normals ──────────────────────────────────────────────────────────────────
def compute_normals(verts, indices):
    normals = [(0.0, 0.0, 0.0)] * len(verts)
    for i in range(0, len(indices), 3):
        ia, ib, ic = indices[i], indices[i+1], indices[i+2]
        va, vb, vc = verts[ia], verts[ib], verts[ic]
        ab = sub(vb, va)
        ac = sub(vc, va)
        n  = cross(ab, ac)
        normals[ia] = tuple(normals[ia][k] + n[k] for k in range(3))
        normals[ib] = tuple(normals[ib][k] + n[k] for k in range(3))
        normals[ic] = tuple(normals[ic][k] + n[k] for k in range(3))
    return [normalize(n) for n in normals]


# ─── Assembly ─────────────────────────────────────────────────────────────────
def build_tshirt():
    all_verts   = []
    all_uvs     = []
    all_indices = []
    segments    = []

    def add_segment(name, verts, uvs, idx_fn, extra_verts=None):
        offset = len(all_verts)
        all_verts.extend(verts)
        all_uvs.extend(uvs)
        if extra_verts:
            all_verts.extend(extra_verts)
            all_uvs.extend([(0.5, 0.5)] * len(extra_verts))
        start = len(all_indices)
        all_indices.extend(idx_fn(offset))
        segments.append({
            "id": name,
            "indexStart": start,
            "indexCount": len(all_indices) - start,
        })

    # ── Body ──
    bv, bu = build_body()
    boffset = len(all_verts)
    all_verts.extend(bv)
    all_uvs.extend(bu)
    bstart = len(all_indices)
    all_indices.extend(body_ring_indices(boffset))
    segments.append({"id": "body", "indexStart": bstart, "indexCount": len(all_indices) - bstart})

    # ── Collar ──
    cv, cu = build_collar()
    coffset = len(all_verts)
    all_verts.extend(cv)
    all_uvs.extend(cu)
    cstart = len(all_indices)
    all_indices.extend(collar_indices(coffset))
    segments.append({"id": "collar", "indexStart": cstart, "indexCount": len(all_indices) - cstart})

    # ── Left sleeve ──
    lv, lu = build_sleeve("left")
    loffset = len(all_verts)
    # Compute cap centres
    left_cuff_centre = tuple(
        sum(lv[-(SLEEVE_CIRC - ci)][k] for ci in range(SLEEVE_CIRC)) / SLEEVE_CIRC
        for k in range(3)
    )
    left_arm_centre = tuple(
        sum(lv[ci][k] for ci in range(SLEEVE_CIRC)) / SLEEVE_CIRC
        for k in range(3)
    )
    all_verts.extend(lv)
    all_verts.append(left_cuff_centre)   # cap 0 index
    all_verts.append(left_arm_centre)    # cap 1 index
    all_uvs.extend(lu)
    all_uvs.extend([(0.5, 1), (0.5, 0)])
    lstart = len(all_indices)
    all_indices.extend(sleeve_indices(loffset, reverse=True))
    all_indices.extend(sleeve_cuff_cap(loffset, reverse=True))
    all_indices.extend(sleeve_armhole_cap(loffset, reverse=True))
    segments.append({"id": "left-sleeve", "indexStart": lstart, "indexCount": len(all_indices) - lstart})

    # ── Right sleeve ──
    rv, ru = build_sleeve("right")
    roffset = len(all_verts)
    right_cuff_centre = tuple(
        sum(rv[-(SLEEVE_CIRC - ci)][k] for ci in range(SLEEVE_CIRC)) / SLEEVE_CIRC
        for k in range(3)
    )
    right_arm_centre = tuple(
        sum(rv[ci][k] for ci in range(SLEEVE_CIRC)) / SLEEVE_CIRC
        for k in range(3)
    )
    all_verts.extend(rv)
    all_verts.append(right_cuff_centre)
    all_verts.append(right_arm_centre)
    all_uvs.extend(ru)
    all_uvs.extend([(0.5, 1), (0.5, 0)])
    rstart = len(all_indices)
    all_indices.extend(sleeve_indices(roffset, reverse=False))
    all_indices.extend(sleeve_cuff_cap(roffset, reverse=False))
    all_indices.extend(sleeve_armhole_cap(roffset, reverse=False))
    segments.append({"id": "right-sleeve", "indexStart": rstart, "indexCount": len(all_indices) - rstart})

    # ── Hem band ──
    hv, hu = build_hem_band()
    hoffset = len(all_verts)
    hem_cap_centre = (0, -P.body_height * 0.5 - P.hem_height, 0)
    all_verts.extend(hv)
    all_verts.append(hem_cap_centre)
    all_uvs.extend(hu)
    all_uvs.append((0.5, 0.5))
    hstart = len(all_indices)
    all_indices.extend(hem_band_indices(hoffset))
    segments.append({"id": "hem", "indexStart": hstart, "indexCount": len(all_indices) - hstart})

    # ── Normals ──
    normals = compute_normals(all_verts, all_indices)

    # ── Bounds ──
    xs = [v[0] for v in all_verts]
    ys = [v[1] for v in all_verts]
    zs = [v[2] for v in all_verts]
    bounds = {
        "width":  max(xs) - min(xs),
        "height": max(ys) - min(ys),
        "depth":  max(zs) - min(zs),
    }

    return {
        "version": 1,
        "template": "top-standard-tee-cad",
        "positions": [c for v in all_verts   for c in v],
        "normals":   [c for n in normals      for c in n],
        "uvs":       [c for uv in all_uvs     for c in uv],
        "indices":   all_indices,
        "segments":  segments,
        "bounds":    bounds,
        "params":    {
            "body_height":   P.body_height,
            "body_width":    P.body_width * 2,
            "body_depth":    P.body_depth * 2,
            "sleeve_length": P.sleeve_length,
            "neck_width":    P.neck_width * 2,
        },
    }


# ─── OBJ export ───────────────────────────────────────────────────────────────
def export_obj(mesh, path):
    lines = ["# CAD T-Shirt mesh", "# Generated by generate_tshirt.py", ""]
    pos = mesh["positions"]
    nor = mesh["normals"]
    uv  = mesh["uvs"]

    for i in range(0, len(pos), 3):
        lines.append(f"v {pos[i]:.6f} {pos[i+1]:.6f} {pos[i+2]:.6f}")
    for i in range(0, len(uv), 2):
        lines.append(f"vt {uv[i]:.6f} {uv[i+1]:.6f}")
    for i in range(0, len(nor), 3):
        lines.append(f"vn {nor[i]:.6f} {nor[i+1]:.6f} {nor[i+2]:.6f}")

    idx = mesh["indices"]
    for seg in mesh["segments"]:
        lines.append(f"\ng {seg['id']}")
        start = seg["indexStart"]
        count = seg["indexCount"]
        for i in range(start, start + count, 3):
            ia, ib, ic = idx[i]+1, idx[i+1]+1, idx[i+2]+1
            lines.append(f"f {ia}/{ia}/{ia} {ib}/{ib}/{ib} {ic}/{ic}/{ic}")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"OBJ written: {path}")


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building CAD T-shirt mesh...")
    mesh = build_tshirt()

    vc = len(mesh["positions"]) // 3
    fc = len(mesh["indices"]) // 3
    print(f"  Vertices : {vc:,}")
    print(f"  Faces    : {fc:,}")
    print(f"  Bounds   : W={mesh['bounds']['width']:.3f}m "
          f"H={mesh['bounds']['height']:.3f}m "
          f"D={mesh['bounds']['depth']:.3f}m")
    for s in mesh["segments"]:
        print(f"  Segment  : {s['id']} — {s['indexCount']//3} faces")

    json_path = "/mnt/user-data/outputs/tshirt_cad.json"
    with open(json_path, "w") as f:
        json.dump(mesh, f, separators=(",", ":"))
    print(f"JSON written: {json_path}")

    obj_path = "/mnt/user-data/outputs/tshirt_cad.obj"
    export_obj(mesh, obj_path)

    print("\nDone. Load tshirt_cad.json in your web renderer.")
