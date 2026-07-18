import json
with open("/home/xiso/IsaacLab/eval_scenes/scene_0_graph.json") as f:
    d = json.load(f)
for nid, node in d["nodes"].items():
    cn = node["class_name"]
    print(cn)  # repr покажет скрытые символы, юникод, нестандартные пробелы