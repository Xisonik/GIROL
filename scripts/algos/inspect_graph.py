# -*- coding: utf-8 -*-
"""Inspect the scene graph the encoder builds from a graph observation.

Capture a REAL observation during any run (works with either encoder — the
graph obs is the same; the dump hook lives in the non-metric encoder):
    GIROL_NONMETRIC=1 GIROL_DUMP_GRAPH=1  <your usual train command>
    # wait for  "[dump] saved graph_flat ..."  then Ctrl-C

Then inspect it offline (no Isaac Sim needed), from the repo root:
    /home/rizo/miniconda3/envs/isaaclab45/bin/python scripts/algos/inspect_graph.py [dump.pt] [env_idx]

It decodes graph_flat[B, 6*M] exactly like NonMetricGraphEncoder:
    per object [object_id, active, is_goal, x, y, z]
    room  = (x<0) + 2*(y<0)      # quadrant
    x-dir = sign(dx) at 0.4 m  (left / align / right)
    y-dir = sign(dy) at 0.4 m  (back / align / front)
and prints a table + saves logs/scene_graph_env<e>.png so you can eyeball it
against the real scene (rooms in quadrants, ★ = goal, lines = goal-star edges).
"""
import os
import sys
import torch

EMB = "source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/text_embeddings.pt"
THR = 0.4
ROOM = {0: "R/F", 1: "L/F", 2: "R/B", 3: "L/B"}   # (x>=0/x<0, y>=0/y<0)


def _names():
    p = torch.load(EMB, map_location="cpu")
    return {int(k): v for k, v in p.get("object_id_to_name", {}).items()}


def _xdir(d):
    return "left " if d < -THR else ("right" if d > THR else "align")


def _ydir(d):
    return "back " if d < -THR else ("front" if d > THR else "align")


def analyze(gf, e):
    M = gf.shape[1] // 6
    g = gf.view(gf.shape[0], M, 6)[e]
    return {
        "M": M,
        "oid": g[:, 0].long(),
        "active": g[:, 1],
        "is_goal": g[:, 2],
        "x": g[:, 3],
        "y": g[:, 4],
        "room": (g[:, 3] < 0).long() + 2 * (g[:, 4] < 0).long(),
        "gi": int(g[:, 2].argmax()),
    }


def print_table(info, names, e):
    gi, oid, x, y, room = info["gi"], info["oid"], info["x"], info["y"], info["room"]
    active = info["active"]
    print(f"\n=== env {e}: {info['M']} objects | goal = #{gi} "
          f"{names.get(int(oid[gi]), '?')} in room {ROOM[int(room[gi])]} ===")
    # room occupancy (ACTIVE objects per quadrant) -> verifies the 4-room scene
    occ = {ROOM[r]: sum(1 for i in range(info["M"]) if active[i] > 0.5 and int(room[i]) == r)
           for r in range(4)}
    n_used = sum(1 for v in occ.values() if v > 0)
    print(f"    room occupancy (active objs): {occ}   -> {n_used}/4 rooms populated")
    print(f"{'#':>2} {'name':<12} {'act':>3} {'gl':>2} {'x':>7} {'y':>7} {'room':>4} | dir vs goal (x, y, room)")
    for i in range(info["M"]):
        nm = names.get(int(oid[i]), f"id{int(oid[i])}")
        if i == gi:
            rel = "(goal)"
        else:
            rel = f"{_xdir(float(x[i] - x[gi]))}, {_ydir(float(y[i] - y[gi]))}, " \
                  f"{'same' if int(room[i]) == int(room[gi]) else 'diff'}"
        print(f"{i:>2} {nm:<12} {int(info['active'][i]):>3} {int(info['is_goal'][i]):>2} "
              f"{float(x[i]):>7.2f} {float(y[i]):>7.2f} {ROOM[int(room[i])]:>4} | {rel}")


def print_tree(info, names, e):
    """Hierarchical view: Scene -> Room (quadrant) -> objects (as the room-aware graph groups them)."""
    gi, oid, x, y, room, active = (info["gi"], info["oid"], info["x"],
                                   info["y"], info["room"], info["active"])
    goal_room = int(room[gi])
    print(f"\nSCENE  env {e}  ({info['M']} objects, goal = "
          f"{names.get(int(oid[gi]), '?')})")
    for r in range(4):
        members = [i for i in range(info["M"])
                   if int(room[i]) == r and (active[i] > 0.5 or i == gi)]
        last_room = (r == 3)
        rconn, vbar = ("└──", "    ") if last_room else ("├──", "│   ")
        gtag = "   <-- GOAL ROOM" if r == goal_room else ""
        print(f"{rconn} Room {ROOM[r]:<4} [{len(members)} obj]{gtag}")
        for j, i in enumerate(members):
            oconn = "└──" if j == len(members) - 1 else "├──"
            mark = "* " if i == gi else "  "
            nm = names.get(int(oid[i]), f"id{int(oid[i])}")
            note = " [GOAL]" if i == gi else ""
            print(f"{vbar}{oconn} {mark}{nm:<12} ({float(x[i]):>6.2f}, {float(y[i]):>6.2f}){note}")


def plot(info, names, e, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {0: "tab:blue", 1: "tab:green", 2: "tab:orange", 3: "tab:red"}
    gi, oid, x, y, room, active = (info["gi"], info["oid"], info["x"],
                                   info["y"], info["room"], info["active"])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axhline(0, color="gray", lw=1); ax.axvline(0, color="gray", lw=1)
    xg, yg = float(x[gi]), float(y[gi])
    for i in range(info["M"]):
        if i != gi and active[i] < 0.5:
            continue
        xi, yi = float(x[i]), float(y[i])
        if i == gi:
            ax.scatter(xi, yi, s=500, marker="*", color="black", zorder=6)
        else:
            ax.plot([xg, xi], [yg, yi], color=col[int(room[i])], lw=0.6, alpha=0.4, zorder=1)
            ax.scatter(xi, yi, s=130, color=col[int(room[i])], zorder=4)
        ax.annotate(names.get(int(oid[i]), str(int(oid[i]))), (xi, yi),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_title(f"env {e}: scene graph  (quadrant rooms, ★ = goal, lines = goal-star)")
    ax.set_xlabel("x  (right +)"); ax.set_ylabel("y  (front +)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[plot] {out}")


def main():
    dump = sys.argv[1] if len(sys.argv) > 1 else "logs/scene_dump.pt"
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
    envs = [int(sys.argv[2])] if len(sys.argv) > 2 else list(range(min(3, gf.shape[0])))
    for e in envs:
        info = analyze(gf, e)
        print_table(info, names, e)
        print_tree(info, names, e)
        plot(info, names, e, f"logs/scene_graph_env{e}.png")


if __name__ == "__main__":
    main()
