#!/usr/bin/env python3
"""
track_builder.py — Standalone track generator (rails + sleepers) for 3D scenes

Matches the style used in your original script:
- Rails:  face '#555555', edge '#222222', lw=0.45
- Sleepers: face '#b58840', edge '#7a4f1d', lw=0.45
"""
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Primitive

def make_box(length=1.0, width=1.0, height=1.0, center=(0,0,0)):
    L, W, H = length/2.0, width/2.0, height/2.0
    cx, cy, cz = center
    V = np.array([
        [cx - L, cy - W, cz - H],
        [cx + L, cy - W, cz - H],
        [cx + L, cy + W, cz - H],
        [cx - L, cy + W, cz - H],
        [cx - L, cy - W, cz + H],
        [cx + L, cy - W, cz + H],
        [cx + L, cy + W, cz + H],
        [cx - L, cy + W, cz + H],
    ], dtype=float)
    F = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
    return V, F

# Build rails + sleepers

def build_track_solids(track_length, rail_top_z, gauge=1.676, rail_width=0.07, rail_height=0.16,
                        sleeper_len_x=0.25, sleeper_width_y=None, sleeper_thickness=0.12, sleeper_spacing=0.6):
    if sleeper_width_y is None or sleeper_width_y <= 0:
        sleeper_width_y = gauge + 0.6
    y_rail = gauge / 2.0
    z_rail_center = rail_top_z - rail_height/2.0
    z_sleeper_center = z_rail_center - (rail_height/2.0) - (sleeper_thickness/2.0)

    solids = []
    for sign in (-1.0, +1.0):
        V, F = make_box(length=track_length, width=rail_width, height=rail_height,
                        center=(0.0, sign*y_rail, z_rail_center))
        solids.append((V, F, {'color': '#555555', 'alpha': 1.0, 'edgecolor': '#222222', 'lw': 0.45}))
    n_pos = int(np.floor((track_length/2.0) / sleeper_spacing))
    x_positions = np.linspace(-n_pos*sleeper_spacing, n_pos*sleeper_spacing, num=2*n_pos+1)
    for x in x_positions:
        V, F = make_box(length=sleeper_len_x, width=sleeper_width_y, height=sleeper_thickness,
                        center=(x, 0.0, z_sleeper_center))
        solids.append((V, F, {'color': '#b58840', 'alpha': 0.98, 'edgecolor': '#7a4f1d', 'lw': 0.45}))
    return solids


def build_track_under_coach(coach_length, coach_bottom_z, wheel_radius, gauge=1.676,
                             rail_width=0.07, rail_height=0.16, sleeper_len_x=0.25, sleeper_width_y=None,
                             sleeper_thickness=0.12, sleeper_spacing=0.6, length_factor=1.6):
    rail_top_z = coach_bottom_z - wheel_radius
    track_len = coach_length * float(length_factor)
    return build_track_solids(track_len, rail_top_z, gauge, rail_width, rail_height,
                              sleeper_len_x, sleeper_width_y, sleeper_thickness, sleeper_spacing)

# Draw helper

def add_solids(ax, solids):
    for (V, F, st) in solids:
        polys = [[V[i] for i in face] for face in F]
        coll = Poly3DCollection(polys,
                                facecolors=st.get('color', '#cccccc'),
                                edgecolor=st.get('edgecolor', 'k'),
                                linewidths=st.get('lw', 0.6),
                                alpha=st.get('alpha', 1.0))
        ax.add_collection3d(coll)
    return ax

