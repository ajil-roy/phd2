#!/usr/bin/env python3
"""
main_rotator.py — Rotate a 3D coach (yaw/pitch/roll) with grids, hatch, themes & track.
Adds CLI flags: --panel_eps (avoid z-fighting on doors/windows), --floor_color (dark floor).
Patched: --rear_x to draw an 'X' on the REAR face.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from rotating_coach import TrainCoach3D, compose_rpy
from track_builder import build_track_under_coach, add_solids

# ------------------ Theme palettes ------------------
def get_theme(name: str):
    name = (name or 'none').lower()
    if name == 'rajdhani':
        return dict(body_color="#9B1C1C", roof_color="#7A7D82", front_color="#D9D9D2",
                    rear_color="#6B7280", glass_color="#0C1A25", door_color="#E6E6E6",
                    grid_color="#5A5A5A", wheel_color="#2F2F2F", axle_color="#888888",
                    floor_color="#2F2F2F")
    if name == 'shatabdi':
        return dict(body_color="#4CA3D9", roof_color="#8A8F94", front_color="#F3E9D2",
                    rear_color="#9CA3AF", glass_color="#0C1A25", door_color="#E6E6E6",
                    grid_color="#6B7280", wheel_color="#2F2F2F", axle_color="#888888",
                    floor_color="#3A3A3A")
    if name == 'tejas':
        return dict(body_color="#F59E0B", roof_color="#D6D6D6", front_color="#F3E9D2",
                    rear_color="#9CA3AF", glass_color="#0C1A25", door_color="#E6E6E6",
                    grid_color="#7A7A7A", wheel_color="#2F2F2F", axle_color="#888888",
                    floor_color="#2E2E2E")
    if name == 'duronto':
        return dict(body_color="#D1F28E", roof_color="#7A7D82", front_color="#F3E9D2",
                    rear_color="#9CA3AF", glass_color="#0C1A25", door_color="#E6E6E6",
                    grid_color="#6B7280", wheel_color="#2F2F2F", axle_color="#888888",
                    floor_color="#2F2F2F")
    if name == 'vb_white':
        return dict(body_color="#FFFFFF", roof_color="#D6D6D6", front_color="#F3E9D2",
                    rear_color="#9CA3AF", glass_color="#0C1A25", door_color="#E6E6E6",
                    grid_color="#5A5A5A", wheel_color="#2F2F2F", axle_color="#888888",
                    floor_color="#2F2F2F")
    if name == 'vb_orange':
        return dict(body_color="#F97316", roof_color="#9CA3AF", front_color="#F3E9D2",
                    rear_color="#6B7280", glass_color="#0C1A25", door_color="#E6E6E6",
                    grid_color="#5A5A5A", wheel_color="#2F2F2F", axle_color="#888888",
                    floor_color="#2F2F2F")
    return None

# ------------------ CLI ------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Rotate a 3D coach (yaw/pitch/roll) with grids, hatch, themes & track")
    ap.add_argument("--yaw", type=float, default=0.0, help="Yaw (deg) about Z")
    ap.add_argument("--pitch", type=float, default=0.0, help="Pitch (deg) about Y")
    ap.add_argument("--roll", type=float, default=0.0, help="Roll (deg) about X")
    ap.add_argument("--order", type=str, default="zyx", help="Rotation order, e.g., 'zyx' (Yaw->Pitch->Roll)")

    # Themes
    ap.add_argument("--theme", type=str, default="none",
                    choices=["none","rajdhani","shatabdi","tejas","duronto","vb_white","vb_orange"],
                    help="Apply a pre-set livery theme (overrides per-color flags)")

    # Coach dimensions & base colors (used only when --theme none)
    ap.add_argument("--length", type=float, default=23.54, help="Coach body length")
    ap.add_argument("--width", type=float, default=3.24, help="Coach width")
    ap.add_argument("--height", type=float, default=4.04, help="Coach height")
    ap.add_argument("--body_color", type=str, default="#F2A922", help="Body (side) color")
    ap.add_argument("--roof_color", type=str, default="#D6D6D6", help="Roof color")
    ap.add_argument("--glass_color", type=str, default="#0C1A25", help="Window glass color")
    ap.add_argument("--door_color", type=str, default="#E6E6E6", help="Door color")
    ap.add_argument("--front_color", type=str, default="#F3E9D2", help="Front (+X) face color")
    ap.add_argument("--rear_color", type=str, default="#9CA3AF", help="Rear (−X) face color")
    ap.add_argument("--floor_color", type=str, default="#3A3A3A", help="Floor (bottom face) color")

    # Grids
    ap.add_argument("--grid_rows", type=int, default=8, help="Grid rows (front/rear)")
    ap.add_argument("--grid_cols", type=int, default=14, help="Grid cols (front/rear)")
    ap.add_argument("--grid_color", type=str, default="#6B7280", help="Grid line color")
    ap.add_argument("--grid_linewidth", type=float, default=0.7, help="Grid line width")
    ap.add_argument("--no_front_grid", action="store_true", help="Disable front grid overlay")
    ap.add_argument("--no_rear_grid", action="store_true", help="Disable rear grid overlay")
    ap.add_argument("--no_bottom_grid", action="store_true", help="Disable bottom grid overlay")

    # Diagonal hatch
    ap.add_argument("--hatch_front", action="store_true", help="Enable diagonal hatch on front")
    ap.add_argument("--hatch_rear", action="store_true", help="Enable diagonal hatch on rear")
    ap.add_argument("--hatch_bottom", action="store_true", help="Enable diagonal hatch on bottom")
    ap.add_argument("--hatch_spacing", type=float, default=0.35, help="Diagonal hatch spacing (m)")
    ap.add_argument("--hatch_color", type=str, default="#7A7A7A", help="Diagonal hatch color")
    ap.add_argument("--hatch_linewidth", type=float, default=0.8, help="Diagonal hatch line width")
    ap.add_argument("--hatch_style", type=str, default='diag', choices=['diag','back'], help="'diag' (/) or 'back' (\\)")
    ap.add_argument("--hatch_cross", action="store_true", help="Draw both '/' and '\\'")

    # NEW: Rear X (two diagonals)
    ap.add_argument("--rear_x", action="store_true", help="Draw an 'X' on the REAR face (top-left↘︎bottom-right and top-right↙︎bottom-left)")
    ap.add_argument("--rear_x_color", type=str, default="#2F2F2F", help="Rear 'X' color")
    ap.add_argument("--rear_x_linewidth", type=float, default=1.5, help="Rear 'X' line width")

    # Feature toggles
    ap.add_argument("--no_doors", action="store_true", help="Disable doors panels")
    ap.add_argument("--no_windows", action="store_true", help="Disable windows panels")
    ap.add_argument("--no_wheels", action="store_true", help="Disable wheels/bogies")

    # Panel epsilon factor (scaled by max(L,W,H))
    ap.add_argument("--panel_eps", type=float, default=1e-3,
                    help="Panel epsilon factor (absolute epsilon = panel_eps * max(L,W,H). Use 0 to disable.")

    # Track
    ap.add_argument("--show_track", action="store_true", help="Render track (rails + sleepers) under the coach")
    ap.add_argument("--no_show_track", dest="show_track", action="store_false", help="Disable track rendering")
    ap.set_defaults(show_track=False)
    ap.add_argument("--track_follow_coach", action="store_true", help="Rotate track with same yaw/pitch/roll as coach")
    ap.add_argument("--no_track_follow_coach", dest="track_follow_coach", action="store_false", help="Keep track flat (no rotation)")
    ap.set_defaults(track_follow_coach=True)
    ap.add_argument("--track_gauge", type=float, default=-1.0, help="Track gauge; <=0 to use coach's gauge")
    ap.add_argument("--rail_width", type=float, default=0.07, help="Rail head width")
    ap.add_argument("--rail_height", type=float, default=0.16, help="Rail height")
    ap.add_argument("--sleeper_len_x", type=float, default=0.25, help="Sleeper length along X")
    ap.add_argument("--sleeper_width", type=float, default=-1.0, help="Sleeper width across Y; <=0 -> (gauge+0.6)")
    ap.add_argument("--sleeper_thk", type=float, default=0.12, help="Sleeper thickness (Z)")
    ap.add_argument("--sleeper_spacing", type=float, default=0.6, help="Sleeper spacing along X")
    ap.add_argument("--track_len_factor", type=float, default=1.6, help="Track length factor relative to coach length")
    return ap.parse_args()


def rotate_solids(solids, R):
    rotated = []
    for (V, F, st) in solids:
        V_rot = (R @ V.T).T
        rotated.append((V_rot, F, st))
    return rotated


def main():
    args = parse_args()
    theme = get_theme(args.theme)
    if theme:
        colors = theme
    else:
        colors = dict(body_color=args.body_color, roof_color=args.roof_color, glass_color=args.glass_color,
                      door_color=args.door_color, front_color=args.front_color, rear_color=args.rear_color,
                      grid_color=args.grid_color, floor_color=args.floor_color)
    colors.setdefault('wheel_color', '#2F2F2F')
    colors.setdefault('axle_color', '#888888')

    coach = TrainCoach3D(
        length=args.length,
        width=args.width,
        height=args.height,
        body_color=colors['body_color'],
        roof_color=colors['roof_color'],
        glass_color=colors['glass_color'],
        door_color=colors['door_color'],
        front_color=colors['front_color'],
        rear_color=colors['rear_color'],
        floor_color=colors.get('floor_color', args.floor_color),
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        grid_color=colors['grid_color'],
        grid_linewidth=args.grid_linewidth,
        front_grid=(not args.no_front_grid),
        rear_grid= (not args.no_rear_grid),
        bottom_grid=(not args.no_bottom_grid),
        # diagonal hatch
        hatch_front=args.hatch_front,
        hatch_rear=args.hatch_rear,
        hatch_bottom=args.hatch_bottom,
        hatch_spacing=args.hatch_spacing,
        hatch_color=args.hatch_color,
        hatch_linewidth=args.hatch_linewidth,
        hatch_style=args.hatch_style,
        hatch_cross=args.hatch_cross,
        # features
        add_doors=(not args.no_doors),
        add_windows=(not args.no_windows),
        add_wheels=(not args.no_wheels),
        # wheels
        wheel_color=colors.get('wheel_color', '#2F2F2F'),
        axle_color=colors.get('axle_color', '#888888'),
        # panel epsilon factor
        panel_eps_factor=args.panel_eps,
        # NEW: rear X settings
        rear_x=(not args.rear_x),
        rear_x_color=args.rear_x_color,
        rear_x_linewidth=args.rear_x_linewidth,
    )

    # Render coach
    fig, ax = coach.render(yaw=args.yaw, pitch=args.pitch, roll=args.roll, order=args.order)

    # Track
    if args.show_track:
        coach_bottom_z = - coach.height / 2.0
        wheel_radius = coach.wheels.get('wheel_radius', 0.46)
        gauge = args.track_gauge if args.track_gauge > 0 else coach.wheels.get('gauge', 1.676)
        sleeper_width = None if args.sleeper_width <= 0 else args.sleeper_width
        track_solids = build_track_under_coach(coach_length=coach.length,
                                               coach_bottom_z=coach_bottom_z,
                                               wheel_radius=wheel_radius,
                                               gauge=gauge,
                                               rail_width=args.rail_width,
                                               rail_height=args.rail_height,
                                               sleeper_len_x=args.sleeper_len_x,
                                               sleeper_width_y=sleeper_width,
                                               sleeper_thickness=args.sleeper_thk,
                                               sleeper_spacing=args.sleeper_spacing,
                                               length_factor=args.track_len_factor)
        if args.track_follow_coach:
            R = compose_rpy(args.yaw, args.pitch, args.roll, order=args.order)
            track_solids = rotate_solids(track_solids, R)
        add_solids(ax, track_solids)

    plt.show()

if __name__ == "__main__":
    main()
