#!/usr/bin/env python3
"""
rotating_coach.py — 3D coach with wheels, face colors, GRID + DIAGONAL HATCH overlays
Enhanced: always-on 'X' on the REAR (-X) face by default (can be disabled via flag if wired).
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ------------------ Rotation matrices ------------------
def Rx(deg):
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],[0, c, -s],[0, s, c]], dtype=float)

def Ry(deg):
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[ c, 0, s],[ 0, 1, 0],[-s, 0, c]], dtype=float)

def Rz(deg):
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],[s,  c, 0],[0,  0, 1]], dtype=float)

def compose_rpy(yaw, pitch, roll, order="zyx"):
    mats = {'x': Rx(roll), 'y': Ry(pitch), 'z': Rz(yaw)}
    R = np.eye(3)
    for axis in order:
        R = mats[axis] @ R
    return R

# ------------------ Geometry primitives ------------------
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
    F = [
        [0,1,2,3], # bottom (z=-H)
        [4,5,6,7], # top (z=+H)
        [0,1,5,4], # -Y side
        [1,2,6,5], # +X side (FRONT)
        [2,3,7,6], # +Y side
        [3,0,4,7], # -X side (REAR)
    ]
    return V, F

# Cylinder along Y (for axles/wheels)
def make_cylinder_y(center=(0,0,0), radius=0.5, length=0.2, n=36):
    cx, cy, cz = center
    y0, y1 = cy - length/2.0, cy + length/2.0
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    ring0 = np.array([[cx + radius*np.cos(t), y0, cz + radius*np.sin(t)] for t in th])
    ring1 = np.array([[cx + radius*np.cos(t), y1, cz + radius*np.sin(t)] for t in th])
    V = np.vstack([ring0, ring1])
    F = []
    for i in range(n):
        j = (i + 1) % n
        F.append([i, j, j + n, i + n])
    F.append(list(range(0, n)))
    F.append(list(range(2*n-1, n-1, -1)))
    return V, F

# ------------------ Plot helpers ------------------
def add_mesh(ax, V, F, color='#cccccc', edgecolor='k', lw=0.6, alpha=1.0):
    polys = [[V[i] for i in face] for face in F]
    if len(polys) == 0:
        return
    coll = Poly3DCollection(polys, facecolors=color, edgecolor=edgecolor, linewidths=lw, alpha=alpha)
    ax.add_collection3d(coll)

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d(); y_limits = ax.get_ylim3d(); z_limits = ax.get_zlim3d()
    ranges = [abs(x_limits[1]-x_limits[0]), abs(y_limits[1]-y_limits[0]), abs(z_limits[1]-z_limits[0])]
    max_range = max(ranges)
    x_mid = np.mean(x_limits); y_mid = np.mean(y_limits); z_mid = np.mean(z_limits)
    ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
    ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
    ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])

# ------------------ Grid overlay helpers (LINES) ------------------
def add_face_grid_lines(lines, L, W, H, face, rows, cols, color, lw):
    eps = 1e-3 * max(L, W, H)
    if face in ('front','rear'):
        x = (+L/2.0 + eps) if face=='front' else (-L/2.0 - eps)
        y_edges = np.linspace(-W/2.0, +W/2.0, cols+1)
        z_edges = np.linspace(-H/2.0, +H/2.0, rows+1)
        for y in y_edges:
            P = np.array([[x, y, -H/2.0],[x, y, +H/2.0]], dtype=float)
            lines.append((P, {'color': color, 'lw': lw}))
        for z in z_edges:
            P = np.array([[x, -W/2.0, z],[x, +W/2.0, z]], dtype=float)
            lines.append((P, {'color': color, 'lw': lw}))
    elif face == 'bottom':
        z = -H/2.0 - eps
        x_edges = np.linspace(-L/2.0, +L/2.0, cols+1)
        y_edges = np.linspace(-W/2.0, +W/2.0, rows+1)
        for y in y_edges:
            P = np.array([[-L/2.0, y, z],[+L/2.0, y, z]], dtype=float)
            lines.append((P, {'color': color, 'lw': lw}))
        for x in x_edges:
            P = np.array([[x, -W/2.0, z],[x, +W/2.0, z]], dtype=float)
            lines.append((P, {'color': color, 'lw': lw}))

# ------------------ Rear 'X' helper ------------------
def add_rear_x_lines(lines, L, W, H, color, lw):
    """Add two diagonals on the REAR (-X) face: TL→BR and TR→BL."""
    eps = 1e-3 * max(L, W, H)
    x = -L/2.0 - eps
    P1 = np.array([[x, -W/2.0, +H/2.0],[x, +W/2.0, -H/2.0]], dtype=float)  # TL→BR
    P2 = np.array([[x, +W/2.0, +H/2.0],[x, -W/2.0, -H/2.0]], dtype=float)  # TR→BL
    lines.append((P1, {'color': color, 'lw': lw}))
    lines.append((P2, {'color': color, 'lw': lw}))

# ------------------ Diagonal hatch helpers (LINES) ------------------

def _clip_diag_segment(ymin, ymax, zmin, zmax, c, diag=True):
    pts = []
    if diag:
        z_at_ymin = ymin - c
        if zmin <= z_at_ymin <= zmax:
            pts.append((ymin, z_at_ymin))
        z_at_ymax = ymax - c
        if zmin <= z_at_ymax <= zmax:
            pts.append((ymax, z_at_ymax))
        y_at_zmin = c + zmin
        if ymin <= y_at_zmin <= ymax:
            pts.append((y_at_zmin, zmin))
        y_at_zmax = c + zmax
        if ymin <= y_at_zmax <= ymax:
            pts.append((y_at_zmax, zmax))
    else:
        z_at_ymin = c - ymin
        if zmin <= z_at_ymin <= zmax:
            pts.append((ymin, z_at_ymin))
        z_at_ymax = c - ymax
        if zmin <= z_at_ymax <= zmax:
            pts.append((ymax, z_at_ymax))
        y_at_zmin = c - zmin
        if ymin <= y_at_zmin <= ymax:
            pts.append((y_at_zmin, zmin))
        y_at_zmax = c - zmax
        if ymin <= y_at_zmax <= ymax:
            pts.append((y_at_zmax, zmax))
    if len(pts) >= 2:
        p0 = pts[0]
        far = max(pts[1:], key=lambda p: (p[0]-p0[0])**2 + (p[1]-p0[1])**2)
        return p0, far
    return None

def add_face_hatch_lines(lines, L, W, H, face, spacing, color, lw, style='diag', both=False):
    eps = 1e-3 * max(L, W, H)
    if face in ('front','rear'):
        x = (+L/2.0 + eps) if face=='front' else (-L/2.0 - eps)
        ymin, ymax = -W/2.0, +W/2.0
        zmin, zmax = -H/2.0, +H/2.0
        cmin = ymin - zmax
        cmax = ymax - zmin
        cs = np.arange(cmin, cmax + spacing, spacing)
        for c in cs:
            seg = _clip_diag_segment(ymin, ymax, zmin, zmax, c, diag=(style=='diag'))
            if seg is not None:
                (y0,z0),(y1,z1) = seg
                lines.append((np.array([[x,y0,z0],[x,y1,z1]], dtype=float), {'color': color, 'lw': lw}))
        if both:
            cmin2 = ymin + zmin
            cmax2 = ymax + zmax
            cs2 = np.arange(cmin2, cmax2 + spacing, spacing)
            for c in cs2:
                seg = _clip_diag_segment(ymin, ymax, zmin, zmax, c, diag=False)
                if seg is not None:
                    (y0,z0),(y1,z1) = seg
                    lines.append((np.array([[x,y0,z0],[x,y1,z1]], dtype=float), {'color': color, 'lw': lw}))
    elif face=='bottom':
        z = -H/2.0 - eps
        xmin, xmax = -L/2.0, +L/2.0
        ymin, ymax = -W/2.0, +W/2.0
        cmin = xmin - ymax
        cmax = xmax - ymin
        cs = np.arange(cmin, cmax + spacing, spacing)
        for c in cs:
            pts = []
            y_at_xmin = xmin - c
            if ymin <= y_at_xmin <= ymax:
                pts.append((xmin, y_at_xmin))
            y_at_xmax = xmax - c
            if ymin <= y_at_xmax <= ymax:
                pts.append((xmax, y_at_xmax))
            x_at_ymin = c + ymin
            if xmin <= x_at_ymin <= xmax:
                pts.append((x_at_ymin, ymin))
            x_at_ymax = c + ymax
            if xmin <= x_at_ymax <= xmax:
                pts.append((x_at_ymax, ymax))
            if len(pts) >= 2:
                p0 = pts[0]
                far = max(pts[1:], key=lambda p: (p[0]-p0[0])**2 + (p[1]-p0[1])**2)
                (x0,y0) = p0; (x1,y1) = far
                lines.append((np.array([[x0,y0,z],[x1,y1,z]], dtype=float), {'color': color, 'lw': lw}))
        if both:
            cmin2 = xmin + ymin
            cmax2 = xmax + ymax
            cs2 = np.arange(cmin2, cmax2 + spacing, spacing)
            for c in cs2:
                pts = []
                y_at_xmin = c - xmin
                if ymin <= y_at_xmin <= ymax:
                    pts.append((xmin, y_at_xmin))
                y_at_xmax = c - xmax
                if ymin <= y_at_xmax <= ymax:
                    pts.append((xmax, y_at_xmax))
                x_at_ymin = c - ymin
                if xmin <= x_at_ymin <= xmax:
                    pts.append((x_at_ymin, ymin))
                x_at_ymax = c - ymax
                if xmin <= x_at_ymax <= xmax:
                    pts.append((x_at_ymax, ymax))
                if len(pts) >= 2:
                    p0 = pts[0]
                    far = max(pts[1:], key=lambda p: (p[0]-p0[0])**2 + (p[1]-p0[1])**2)
                    (x0,y0) = p0; (x1,y1) = far
                    lines.append((np.array([[x0,y0,z],[x1,y1,z]], dtype=float), {'color': color, 'lw': lw}))

# ------------------ TrainCoach3D ------------------
class TrainCoach3D:
    def __init__(self,
                 length=23.54,
                 width=3.24,
                 height=4.04,
                 body_color="#F2A922",
                 roof_color="#D6D6D6",
                 glass_color="#0C1A25",
                 door_color="#E6E6E6",
                 edge_color="#222222",
                 front_color="#F3E9D2", # cream
                 rear_color="#9CA3AF",  # grey
                 floor_color="#3A3A3A", # DARKER floor (bottom)
                 front_grid=True,
                 rear_grid=True,
                 bottom_grid=True,
                 grid_rows=8,
                 grid_cols=14,
                 grid_color="#6B7280",
                 grid_linewidth=0.7,
                 # Diagonal hatch flags
                 hatch_front=True,
                 hatch_rear=True,
                 hatch_bottom=False,
                 hatch_spacing=0.35,
                 hatch_color="#7A7A7A",
                 hatch_linewidth=0.8,
                 hatch_style='diag', # 'diag' (/) or 'back' (\)
                 hatch_cross=False,
                 add_doors=True,
                 add_windows=True,
                 add_wheels=True,
                 # Doors/windows geometry
                 door_width=1.10,
                 door_height=2.00,
                 door_offset=2.20,
                 door_sill=0.35,
                 win_width=1.10,
                 win_height=0.85,
                 win_gap=0.35,
                 win_sill=1.55,
                 panel_thk=0.03,
                 # Wheel/bogie parameters
                 gauge=1.676,
                 wheel_radius=0.46,
                 wheel_width=0.09,
                 axle_radius=0.05,
                 axle_spacing=2.0,
                 bogie_offset_frac=0.36,
                 clearance=0.10,
                 wheel_color="#2f2f2f",
                 axle_color="#888888",
                 # Panel epsilon factor
                 panel_eps_factor=1e-3,
                 # Rear 'X' (ALWAYS TRUE by default)
                 rear_x=True,
                 rear_x_color="#ff2d20",
                 rear_x_linewidth=1.4,
                 ):
        self.length = float(length)
        self.width = float(width)
        self.height = float(height)
        self.colors = dict(body=body_color, roof=roof_color, glass=glass_color, door=door_color, edge=edge_color,
                           wheel=wheel_color, axle=axle_color, front=front_color, rear=rear_color,
                           floor=floor_color, grid=grid_color, hatch=hatch_color,
                           rear_x=rear_x_color)
        self.flags = dict(add_doors=add_doors, add_windows=add_windows, add_wheels=add_wheels,
                          front_grid=front_grid, rear_grid=rear_grid, bottom_grid=bottom_grid,
                          hatch_front=hatch_front, hatch_rear=hatch_rear, hatch_bottom=hatch_bottom,
                          rear_x=rear_x)
        self.panel = dict(door_width=door_width, door_height=door_height, door_offset=door_offset, door_sill=door_sill,
                          win_width=win_width, win_height=win_height, win_gap=win_gap, win_sill=win_sill,
                          panel_thk=panel_thk, eps_factor=float(panel_eps_factor))
        self.wheels = dict(gauge=gauge, wheel_radius=wheel_radius, wheel_width=wheel_width, axle_radius=axle_radius,
                           axle_spacing=axle_spacing, bogie_offset_frac=bogie_offset_frac, clearance=clearance)
        self.grid = dict(rows=grid_rows, cols=grid_cols, lw=grid_linewidth)
        self.hatch = dict(spacing=hatch_spacing, lw=hatch_linewidth, style=hatch_style, cross=hatch_cross)
        self.rear_x_style = dict(lw=rear_x_linewidth)
        self._solids, self._lines = self._build_geometry()

    def _build_geometry(self):
        solids = []
        lines = []
        L, W, H = self.length, self.width, self.height
        V_body, F_body = make_box(L, W, H, center=(0.0, 0.0, 0.0))
        # Bottom (floor) — DARKER color
        solids.append((V_body.copy(), [F_body[0]],
                      {'color': self.colors['floor'], 'edgecolor': self.colors['edge'], 'alpha': 1.0, 'lw': 0.8}))
        # Top and ±Y sides — body color
        for face_idx in (1, 2, 4):
            face = F_body[face_idx]
            solids.append((V_body.copy(), [face],
                          {'color': self.colors['body'], 'edgecolor': self.colors['edge'], 'alpha': 1.0, 'lw': 0.8}))
        # FRONT (+X) and REAR (−X) — opaque
        solids.append((V_body.copy(), [F_body[3]],
                      {'color': self.colors['front'], 'edgecolor': self.colors['edge'], 'alpha': 1.0, 'lw': 0.9}))
        solids.append((V_body.copy(), [F_body[5]],
                      {'color': self.colors['rear'], 'edgecolor': self.colors['edge'], 'alpha': 1.0, 'lw': 0.9}))

        # Grids as lines
        g = self.grid
        if self.flags['front_grid']:
            add_face_grid_lines(lines, L, W, H, 'front', g['rows'], g['cols'], self.colors['grid'], g['lw'])
        if self.flags['rear_grid']:
            add_face_grid_lines(lines, L, W, H, 'rear', g['rows'], g['cols'], self.colors['grid'], g['lw'])
        if self.flags['bottom_grid']:
            add_face_grid_lines(lines, L, W, H, 'bottom', max(4, g['rows']//2), max(6, g['cols']//2), self.colors['grid'], g['lw'])

        # Rear 'X' (always on by default)
        if self.flags.get('rear_x', True):
            add_rear_x_lines(lines, L, W, H, self.colors['rear_x'], self.rear_x_style['lw'])

        # Hatches as lines
        h = self.hatch
        if self.flags['hatch_front']:
            add_face_hatch_lines(lines, L, W, H, 'front', h['spacing'], self.colors['hatch'], h['lw'],
                                 style=('diag' if h['style']=='diag' else 'back'), both=h['cross'])
        if self.flags['hatch_rear']:
            add_face_hatch_lines(lines, L, W, H, 'rear', h['spacing'], self.colors['hatch'], h['lw'],
                                 style=('diag' if h['style']=='diag' else 'back'), both=h['cross'])
        if self.flags['hatch_bottom']:
            add_face_hatch_lines(lines, L, W, H, 'bottom', h['spacing'], self.colors['hatch'], h['lw'],
                                 style=('diag' if h['style']=='diag' else 'back'), both=h['cross'])

        # Roof slab (thin)
        roof_thk = 0.10
        V_roof, F_roof = make_box(L, W, roof_thk, center=(0.0, 0.0, +H/2.0 - roof_thk/2.0))
        solids.append((V_roof, F_roof, {'color': self.colors['roof'], 'edgecolor': '#777777', 'alpha': 1.0, 'lw': 0.5}))

        # Vestibule panel at +X end
        vestibule_w = 0.90
        V_ves, F_ves = make_box(vestibule_w, W - 0.4, H - 0.6, center=(+L/2.0 - vestibule_w/2.0, 0.0, 0.0))
        solids.append((V_ves, F_ves, {'color': self.colors['door'], 'edgecolor': '#444444', 'alpha': 1.0, 'lw': 0.6}))

        # Doors on sidewalls (±Y), nudged outward by eps
        if self.flags['add_doors']:
            p = self.panel
            eps = p['eps_factor'] * max(L, W, H)
            door_centers_x = (-L/2.0 + p['door_offset'], +L/2.0 - p['door_offset'])
            door_center_z = -H/2.0 + p['door_sill'] + p['door_height']/2.0
            for y_sign in (-1.0, +1.0):
                cy = y_sign * (W/2.0 - p['panel_thk']/2.0 + eps)
                for cx in door_centers_x:
                    Vd, Fd = make_box(length=p['door_width'], width=p['panel_thk'], height=p['door_height'],
                                       center=(cx, cy, door_center_z))
                    solids.append((Vd, Fd, {'color': self.colors['door'], 'edgecolor': self.colors['edge'], 'alpha': 1.0, 'lw': 0.45}))

        # Windows on sidewalls (±Y), nudged outward by eps
        if self.flags['add_windows']:
            p = self.panel
            eps = p['eps_factor'] * max(L, W, H)
            left_block_end = -L/2.0 + p['door_offset'] + p['door_width'] + 0.40
            right_block_start = +L/2.0 - p['door_offset'] - p['door_width'] - 0.40
            span = max(0.5, right_block_start - left_block_end)
            n = max(5, int((span + p['win_gap']) // (p['win_width'] + p['win_gap'])))
            total_w = n * p['win_width']
            gaps = max(0.10, (span - total_w) / max(1, (n - 1)))
            x0 = left_block_end + p['win_width']/2.0
            wz = -H/2.0 + p['win_sill'] + p['win_height']/2.0
            for i in range(n):
                cx = x0 + i * (p['win_width'] + gaps)
                for y_sign in (-1.0, +1.0):
                    cy = y_sign * (W/2.0 - p['panel_thk']/2.0 + eps)
                    Vw, Fw = make_box(length=p['win_width'], width=p['panel_thk'], height=p['win_height'], center=(cx, cy, wz))
                    solids.append((Vw, Fw, {'color': self.colors['glass'], 'edgecolor': '#1a1a1a', 'alpha': 0.98, 'lw': 0.35}))

        # Wheels (bogies + axles + wheels)
        if self.flags['add_wheels']:
            w = self.wheels
            bogie_center_offset = w['bogie_offset_frac'] * L
            wheel_center_z = -H/2.0 - w['clearance'] - w['wheel_radius']
            for bx in (-bogie_center_offset, +bogie_center_offset):
                for axx in (bx - w['axle_spacing']/2.0, bx + w['axle_spacing']/2.0):
                    V_ax, F_ax = make_cylinder_y(center=(axx, 0.0, wheel_center_z), radius=w['axle_radius'], length=w['gauge'], n=28)
                    solids.append((V_ax, F_ax, {'color': self.colors['axle'], 'edgecolor': '#333333', 'alpha': 1.0, 'lw': 0.35}))
                    for y_sign in (-1.0, +1.0):
                        wy = y_sign * (w['gauge']/2.0)
                        V_wh, F_wh = make_cylinder_y(center=(axx, wy, wheel_center_z), radius=w['wheel_radius'], length=w['wheel_width'], n=36)
                        solids.append((V_wh, F_wh, {'color': self.colors['wheel'], 'edgecolor': '#1a1a1a', 'alpha': 1.0, 'lw': 0.35}))
        return solids, lines

    def rotated_solids(self, yaw=0.0, pitch=0.0, roll=0.0, order='zyx'):
        R = compose_rpy(yaw, pitch, roll, order=order)
        rotated = []
        for (V, F, st) in self._solids:
            V_arr = V if isinstance(V, np.ndarray) else np.array(V, dtype=float)
            V_rot = (R @ V_arr.T).T
            rotated.append((V_rot, F, st))
        return rotated

    def rotated_lines(self, yaw=0.0, pitch=0.0, roll=0.0, order='zyx'):
        R = compose_rpy(yaw, pitch, roll, order=order)
        rot_lines = []
        for (P, st) in self._lines:
            P_rot = (R @ P.T).T
            rot_lines.append((P_rot, st))
        return rot_lines

    def render(self, yaw=0.0, pitch=0.0, roll=0.0, order='zyx', figsize=(12,7), dpi=140, elev=18, azim=-55):
        solids = self.rotated_solids(yaw, pitch, roll, order)
        lines = self.rotated_lines(yaw, pitch, roll, order)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"Coach (Rotate about Center) — yaw={yaw:.2f}, pitch={pitch:.2f}, roll={roll:.2f}", fontsize=11)
        for (V, F, st) in solids:
            add_mesh(ax, V, F, color=st.get('color', '#cccccc'), edgecolor=st.get('edgecolor', 'k'),
                     lw=st.get('lw', 0.6), alpha=st.get('alpha', 1.0))
        for (P, st) in lines:
            ax.plot(P[:,0], P[:,1], P[:,2], color=st.get('color', '#6B7280'), linewidth=st.get('lw', 0.9), zorder=50)
        ax.view_init(elev=elev, azim=azim)
        ALL_solids = np.vstack([V if isinstance(V, np.ndarray) else np.array(V) for (V, F, st) in solids])
        ALL_lines = np.vstack([P for (P, st) in lines]) if lines else ALL_solids
        ALL = np.vstack([ALL_solids, ALL_lines]) if lines else ALL_solids
        pad = max(self.length, self.width) * 0.15
        mins = ALL.min(axis=0) - pad
        maxs = ALL.max(axis=0) + pad
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        set_axes_equal(ax)
        ax.grid(False)
        try:
            ax.xaxis.pane.set_visible(False)
            ax.yaxis.pane.set_visible(False)
            ax.zaxis.pane.set_visible(False)
        except Exception:
            pass
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_axis_off()
        plt.tight_layout()
        return fig, ax
# End of module
