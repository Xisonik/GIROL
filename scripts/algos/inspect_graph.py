# -*- coding: utf-8 -*-
"""Inspect the scene graph the encoder builds from a graph observation.

Capture a REAL observation during any run (the graph obs is identical for every
encoder; both graph_encoder.py and hierarchical_graph_encoder.py have the dump hook):
    GIROL_DUMP_GRAPH=1  <your usual train command>
    # wait for  "[dump] saved graph_flat ..."  then Ctrl-C

Then inspect it offline (no Isaac Sim needed), from the repo root:
    python scripts/algos/inspect_graph.py [dump.pt] [env_idx] [--quadrant]

Rooms match HierarchicalGraphEncoder:
  * default: read the sim's layout_rules.json -> exactly len(active_rooms) rooms,
    each object assigned to its NEAREST active room center (faithful to the scene:
    1 room -> 1 node, 2 -> 2, 4 -> 4).
  * --quadrant: the old fixed 4-quadrant binning  room = (x<0) + 2*(y<0).

Prints a table + a Scene->Room->objects tree, and saves logs/scene_graph_env<e>.png.
"""
import os
import sys
import torch

EMB = "source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/text_embeddings.pt"
LAYOUT = "source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/configs/layout_rules.json"
THR = 0.4


def _zone(cx, cy):
    return f"{'R' if cx >= 0 else 'L'}/{'F' if cy >= 0 else 'B'}"


class Rooms:
    """Room model matching the encoder: quadrant (fixed 4) or layout (sim rooms)."""

    def __init__(self, quadrant=False, path=LAYOUT):
        self.quadrant = quadrant or not os.path.exists(path)
        if self.quadrant:
            self.R = 4
            self.centers = None
            self.active = None
            self.labels = ["R/F", "L/F", "R/B", "L/B"]   # (x>=0/x<0, y>=0/y<0)
        else:
            import json
            rl = json.load(open(path))["room_layout"]
            self.active = list(rl["active_rooms"])
            cs = rl["room_centers"]
            self.centers = torch.tensor(
                [[float(cs[r - 1][0]), float(cs[r - 1][1])] for r in self.active])
            self.R = len(self.active)
            self.labels = [_zone(float(c[0]), float(c[1])) for c in self.centers]

    def assign(self, x, y):                       # x,y: [M] -> room id [M]
        if self.quadrant:
            return (x < 0).long() + 2 * (y < 0).long()
        xy = torch.stack([x, y], -1)                                      # [M,2]
        d = (xy.unsqueeze(1) - self.centers.unsqueeze(0)).pow(2).sum(-1)  # [M,R]
        return d.argmin(-1)

    def label(self, r):
        return self.labels[int(r)]


def _names():
    p = torch.load(EMB, map_location="cpu")
    return {int(k): v for k, v in p.get("object_id_to_name", {}).items()}


def _xdir(d):
    return "left " if d < -THR else ("right" if d > THR else "align")


def _ydir(d):
    return "back " if d < -THR else ("front" if d > THR else "align")


def analyze(gf, e, rooms):
    M = gf.shape[1] // 6
    g = gf.view(gf.shape[0], M, 6)[e]
    return {
        "M": M,
        "oid": g[:, 0].long(),
        "active": g[:, 1],
        "is_goal": g[:, 2],
        "x": g[:, 3],
        "y": g[:, 4],
        "room": rooms.assign(g[:, 3], g[:, 4]),
        "gi": int(g[:, 2].argmax()),
    }


def print_table(info, names, e, rooms):
    gi, oid, x, y, room = info["gi"], info["oid"], info["x"], info["y"], info["room"]
    active = info["active"]
    print(f"\n=== env {e}: {info['M']} objects | goal = #{gi} "
          f"{names.get(int(oid[gi]), '?')} in room {rooms.label(room[gi])} ===")
    occ = {rooms.label(r): sum(1 for i in range(info["M"]) if active[i] > 0.5 and int(room[i]) == r)
           for r in range(rooms.R)}
    n_used = sum(1 for v in occ.values() if v > 0)
    print(f"    room occupancy (active objs): {occ}   -> {n_used}/{rooms.R} rooms populated")
    print(f"{'#':>2} {'name':<12} {'act':>3} {'gl':>2} {'x':>7} {'y':>7} {'room':>4} | dir vs goal (x, y, room)")
    for i in range(info["M"]):
        nm = names.get(int(oid[i]), f"id{int(oid[i])}")
        if i == gi:
            rel = "(goal)"
        else:
            rel = f"{_xdir(float(x[i] - x[gi]))}, {_ydir(float(y[i] - y[gi]))}, " \
                  f"{'same' if int(room[i]) == int(room[gi]) else 'diff'}"
        print(f"{i:>2} {nm:<12} {int(active[i]):>3} {int(info['is_goal'][i]):>2} "
              f"{float(x[i]):>7.2f} {float(y[i]):>7.2f} {rooms.label(room[i]):>4} | {rel}")


def print_tree(info, names, e, rooms):
    """Hierarchical view: Scene -> Room -> objects (as the room-aware graph groups them)."""
    gi, oid, x, y, room, active = (info["gi"], info["oid"], info["x"],
                                   info["y"], info["room"], info["active"])
    goal_room = int(room[gi])
    print(f"\nSCENE  env {e}  ({info['M']} objects, goal = "
          f"{names.get(int(oid[gi]), '?')})")
    for r in range(rooms.R):
        members = [i for i in range(info["M"])
                   if int(room[i]) == r and (active[i] > 0.5 or i == gi)]
        last_room = (r == rooms.R - 1)
        rconn, vbar = ("└──", "    ") if last_room else ("├──", "│   ")
        gtag = "   <-- GOAL ROOM" if r == goal_room else ""
        print(f"{rconn} Room {rooms.label(r):<4} [{len(members)} obj]{gtag}")
        for j, i in enumerate(members):
            oconn = "└──" if j == len(members) - 1 else "├──"
            mark = "* " if i == gi else "  "
            nm = names.get(int(oid[i]), f"id{int(oid[i])}")
            note = " [GOAL]" if i == gi else ""
            print(f"{vbar}{oconn} {mark}{nm:<12} ({float(x[i]):>6.2f}, {float(y[i]):>6.2f}){note}")


def plot(info, names, e, out, rooms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple", "tab:brown"]
    col = lambda r: palette[int(r) % len(palette)]
    gi, oid, x, y, room, active = (info["gi"], info["oid"], info["x"],
                                   info["y"], info["room"], info["active"])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axhline(0, color="gray", lw=1); ax.axvline(0, color="gray", lw=1)
    if rooms.centers is not None:                       # mark real room centers
        for r in range(rooms.R):
            cx, cy = float(rooms.centers[r, 0]), float(rooms.centers[r, 1])
            ax.scatter(cx, cy, s=350, marker="s", facecolors="none",
                       edgecolors=col(r), lw=1.5, zorder=2)
            ax.annotate(f"room {rooms.label(r)}", (cx, cy), fontsize=9, color=col(r),
                        xytext=(0, 12), textcoords="offset points", ha="center")
    xg, yg = float(x[gi]), float(y[gi])
    for i in range(info["M"]):
        if i != gi and active[i] < 0.5:
            continue
        xi, yi = float(x[i]), float(y[i])
        if i == gi:
            ax.scatter(xi, yi, s=500, marker="*", color="black", zorder=6)
        else:
            ax.plot([xg, xi], [yg, yi], color=col(room[i]), lw=0.6, alpha=0.4, zorder=1)
            ax.scatter(xi, yi, s=130, color=col(room[i]), zorder=4)
        ax.annotate(names.get(int(oid[i]), str(int(oid[i]))), (xi, yi),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    src = "quadrant (fixed 4)" if rooms.quadrant else f"layout: {rooms.R} sim rooms"
    ax.set_title(f"env {e}: scene graph  ({src}, ★ = goal, lines = goal-star)")
    ax.set_xlabel("x  (right +)"); ax.set_ylabel("y  (front +)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[plot] {out}")


def plot_layered(info, names, e, out, rooms):
    """Node-link view of the hierarchical graph in layers:
    top row = room nodes, bottom = object nodes (grouped under their room),
    edges = goal-star (goal<->object) + containment (object<->room) + room<->room.
    Self-loops (one per node) are omitted for readability.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    palette = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple", "tab:brown"]
    col = lambda r: palette[int(r) % len(palette)]
    gi, oid, room, active = info["gi"], info["oid"], info["room"], info["active"]
    M = info["M"]

    objs = [i for i in range(M) if active[i] > 0.5 or i == gi]      # active objects + goal
    by_room = {r: [i for i in objs if int(room[i]) == r] for r in range(rooms.R)}

    Y_ROOM, Y_OBJ = 2.0, 0.0
    room_x, obj_pos = {}, {}
    x, gap = 0.0, 1.5
    for r in range(rooms.R):
        members = by_room[r]
        xs = [x + k for k in range(max(1, len(members)))]
        for k, i in enumerate(members):
            obj_pos[i] = (xs[k], Y_OBJ)
        room_x[r] = (xs[0] + xs[-1]) / 2.0
        x = xs[-1] + 1 + gap
    if gi in obj_pos:                         # lift the goal so its star fans out
        obj_pos[gi] = (obj_pos[gi][0], 0.95)

    fig, ax = plt.subplots(figsize=(max(12, len(objs) * 0.85), 7.5))

    # edges ---------------------------------------------------------------
    for i in objs:                                                  # containment: object -> room
        ox, oy = obj_pos[i]
        ax.plot([ox, room_x[int(room[i])]], [oy, Y_ROOM], color="0.7", lw=0.7, alpha=0.55, zorder=1)
    for a in range(rooms.R):                                        # room <-> room
        for b in range(a + 1, rooms.R):
            ax.plot([room_x[a], room_x[b]], [Y_ROOM, Y_ROOM], color="tab:purple", lw=1.4, alpha=0.7, zorder=2)
    gx, gy = obj_pos[gi]                                            # goal-star: goal -> every object
    for i in objs:
        if i == gi:
            continue
        ox, oy = obj_pos[i]
        ax.plot([gx, ox], [gy, oy], color="tab:orange", lw=0.6, alpha=0.30, zorder=1)

    # nodes ---------------------------------------------------------------
    for r in range(rooms.R):
        ax.scatter(room_x[r], Y_ROOM, s=2600, marker="s", color=col(r), edgecolors="black", lw=1.2, zorder=4)
        ax.annotate(f"ROOM\n{rooms.label(r)}", (room_x[r], Y_ROOM), ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold", zorder=5)
    for i in objs:
        ox, oy = obj_pos[i]
        if i == gi:
            ax.scatter(ox, oy, s=520, marker="*", color="gold", edgecolors="black", lw=1.0, zorder=6)
        else:
            ax.scatter(ox, oy, s=180, color=col(int(room[i])), edgecolors="black", lw=0.5, zorder=4)
        nm = names.get(int(oid[i]), f"id{int(oid[i])}")
        ax.annotate(nm + (" ★" if i == gi else ""), (ox, oy), rotation=40, ha="right", va="top",
                    fontsize=7, xytext=(-3, -7), textcoords="offset points", zorder=5)

    ax.legend(handles=[
        Line2D([0], [0], color="tab:orange", lw=2, label="goal-star (goal ↔ object)"),
        Line2D([0], [0], color="0.7", lw=2, label="containment (object ↔ room)"),
        Line2D([0], [0], color="tab:purple", lw=2, label="room ↔ room"),
    ], loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.08))
    ax.set_title(f"env {e}: hierarchical graph (layered) — {rooms.R} rooms, "
                 f"{len(objs)} object nodes   (★ = goal; self-loops omitted)")
    ax.set_ylim(-1.0, 2.8); ax.axis("off")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[layered] {out}")


def main():
    argv = sys.argv[1:]
    quadrant = "--quadrant" in argv
    args = [a for a in argv if not a.startswith("--")]
    dump = args[0] if len(args) > 0 else "logs/scene_dump.pt"

    rooms = Rooms(quadrant=quadrant)
    if rooms.quadrant:
        print("[rooms] quadrant (fixed 4): room = (x<0) + 2*(y<0)")
    else:
        print(f"[rooms] layout: {rooms.R} sim rooms {rooms.labels} (active_rooms={rooms.active})")

    names = _names()
    if os.path.exists(dump):
        gf = torch.load(dump, map_location="cpu").float()
        print(f"loaded {dump}: {tuple(gf.shape)}")
    else:
        print(f"[warn] {dump} not found -> synthetic scene (logic check only)")
        M, B = 22, 1
        g = torch.zeros(B, M, 6)
        g[..., 0] = torch.randint(0, 17, (B, M)).float()
        g[..., 1] = 1.0
        g[0, 3, 2] = 1.0
        g[..., 3:5] = torch.randn(B, M, 2) * 4.0
        gf = g.reshape(B, M * 6)
    os.makedirs("logs", exist_ok=True)
    envs = [int(args[1])] if len(args) > 1 else list(range(min(3, gf.shape[0])))
    for e in envs:
        info = analyze(gf, e, rooms)
        print_table(info, names, e, rooms)
        print_tree(info, names, e, rooms)
        plot(info, names, e, f"logs/scene_graph_env{e}.png", rooms)
        plot_layered(info, names, e, f"logs/scene_graph_layered_env{e}.png", rooms)


if __name__ == "__main__":
    main()
