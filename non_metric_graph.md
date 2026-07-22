# Non-Metric Scene Graph for GIROL Orientation

## 1. Goal

The purpose of this representation is to train and evaluate the GIROL orientation module on single-floor indoor scenes containing approximately 3–4 rooms, while removing continuous metric information from the graph.

The current metric baseline uses:

- 22 object nodes;
- a CLIP embedding for each object name;
- node flags such as `active` and `is_goal`;
- continuous object coordinates `x, y, z` inside the node features;
- a fixed star, a fixed chain, and self-loops;
- two GATv2 layers;
- mean pooling;
- a 128-dimensional graph embedding;
- a 36-bin orientation head.

The proposed representation removes:

- `x, y, z`;
- continuous `dx, dy`;
- Euclidean distance;
- numerical bearing or angle;
- metric room dimensions;
- metric path length.

It replaces them with:

- room hierarchy;
- task-conditioned graph topology;
- qualitative directional relations;
- ordinal separation categories;
- edge-aware message passing;
- task-conditioned graph readout.

The main claim to test is:

> Continuous metric coordinates can be replaced by a qualitative, room-aware relational graph while preserving enough spatial information for accurate orientation prediction.

---

## 2. Important distinction

Simply setting:

```python
x = 0
y = 0
z = 0
```

does not produce a meaningful non-metric graph.

If the graph topology remains a fixed star plus an arbitrary chain, then two geometrically different scenes with the same object names and flags may become nearly identical inputs.

The spatial information must therefore move from continuous node coordinates into:

1. graph topology;
2. room membership;
3. qualitative edge relations;
4. ordinal spatial categories.

---

## 3. Scope and assumptions

The intended scenes have:

- one floor;
- approximately 3–4 rooms;
- objects distributed inside the rooms;
- a shared global XY reference frame;
- the scene centre near the origin;
- rooms located in different regions or quadrants of the XY plane.

This design does **not** use door nodes.

It is appropriate when the orientation target is approximately the direct relative orientation from the active object or agent to the goal.

A graph without doors, occupancy, free space, or navigability information cannot always predict the first collision-free heading around walls. It represents qualitative scene layout, not an exact navigation map.

---

# 4. Recommended graph

## 4.1 Representation name

Use a:

> **Room-Aware Qualitative Dual-Star Graph**

The graph contains:

- object nodes;
- room nodes;
- one scene node;
- object-to-room containment edges;
- room-to-room qualitative relations;
- active-centred task edges;
- goal-centred task edges;
- self-loops.

Conceptually:

```text
                         Scene
                    /      |      \
                 Room A  Room B  Room C
                 / | \      |      / | \
              objects    objects   objects

Active object  <--------> relevant objects
Goal object    <--------> relevant objects
```

The room hierarchy captures coarse global structure.

The active and goal stars preserve task-specific local relational information.

---

# 5. Node types

## 5.1 Object nodes

Keep one node for each of the 22 objects.

### Raw object features

```text
CLIP name embedding
active flag
is_goal flag
node-type identifier
```

Do not include:

```text
x
y
z
distance to origin
distance to goal
object yaw
numerical bearing
```

The raw object feature can be written as:

\[
x_i^{obj}
=
[
\operatorname{CLIP}(name_i),
active_i,
goal_i
]
\]

A type-specific object MLP projects the feature into the common GNN hidden dimension.

---

## 5.2 Room nodes

Create one room node for each room in the scene.

### Room features

```text
node type = room
is_current_room
is_goal_room
```

Possible semantic classes:

```text
bedroom
bathroom
kitchen
living_room
hallway
office
unknown
```

Do not expose fixed room IDs such as:

```text
room_0
room_1
room_2
```

as semantic features.

Otherwise, the network may memorize that a particular room index always occupies a particular location.

Room indices should be randomly permuted during training and evaluation.

---

## 5.3 Scene node

A single scene-root node is recommended.

### Scene features

```text
node type = scene
```

The node may use a learned type embedding and carry no scene ID.

The scene node allows information to pass between rooms and gives the encoder a global summary anchor.

---

# 6. Edge topology

## 6.1 Remove the arbitrary chain

The previous fixed chain over object indices should be removed unless node order has an explicit semantic meaning.

An arbitrary chain may:

- introduce meaningless connectivity;
- create artificial long-range paths;
- leak object ordering;
- allow memorization from node index;
- obscure the contribution of the intended qualitative representation.

Keep self-loops, but replace the chain with meaningful relations.

---

## 6.2 Object-to-room hierarchy

For every object node, create two directed edges:

```text
object -> room: inside_room
room -> object: contains_object
```

Example:

```text
chair_3 -> room_SW
room_SW -> chair_3
```

These are structural relations.

They do not need X/Y spatial fields because the corresponding local spatial information can be represented separately.

---

## 6.3 Scene-to-room hierarchy

For every room:

```text
scene -> room: contains_room
room -> scene: inside_scene
```

The `scene -> room` edge may carry a qualitative room position.

### Qualitative room position

Use factorized global room zones:

```text
room_x_zone:
    left
    centre
    right

room_y_zone:
    back
    centre
    front
```

For example:

```text
upper-left room:
    x_zone = left
    y_zone = front

lower-left room:
    x_zone = left
    y_zone = back

lower-right room:
    x_zone = right
    y_zone = back
```

These are categorical relations, not coordinates.

A single combined quadrant category is also possible:

```text
north_west
north_east
south_west
south_east
central
```

The factorized X/Y version is preferred because it generalizes better to non-quadrant layouts.

---

## 6.4 Room-to-room edges

Because the scene contains only 3–4 rooms, connect every ordered pair of rooms.

For every pair:

```text
room_i -> room_j
room_j -> room_i
```

### Room relation fields

```text
x_direction:
    left
    aligned
    right

y_direction:
    behind
    aligned
    front

room_alignment:
    same_row
    same_column
    diagonal
    other

room_connectivity:
    adjacent
    nonadjacent
```

Example:

```text
room_SW -> room_SE:
    x_direction = right
    y_direction = aligned
    room_alignment = same_row
    room_connectivity = adjacent
```

No continuous room centre or inter-room distance is stored.

---

## 6.5 Task-conditioned dual star

The task graph should not always use object node `0` as the hub.

Use:

```text
hub 1 = active object
hub 2 = goal object
```

Connect:

```text
active object <-> all relevant objects
goal object   <-> all relevant objects
```

At minimum, include:

```text
active <-> goal
```

The dual-star structure allows the network to compare active and goal through common landmarks.

A practical version can connect the active and goal objects to:

- every object;
- or only objects in the active and goal rooms;
- or the top-K most stable/relevant landmark objects.

Because there are only 22 objects, connecting to all objects is computationally inexpensive.

---

## 6.6 Self-loops

Retain self-loops for all nodes:

```text
node -> node: self
```

The relation type should explicitly identify a self-loop.

---

# 7. Qualitative spatial edge representation

Each spatial edge represents qualitative X and Y relations between its source and target.

For edge:

```text
i -> j
```

the preprocessing stage may calculate:

```python
dx = x_j - x_i
dy = y_j - y_i
```

These numerical values are used only to generate categorical labels.

They are not included in the saved graph.

---

## 7.1 X direction

```text
x_direction:
    left
    aligned_x
    right
```

Possible rule:

```python
if dx < -alignment_threshold:
    x_direction = LEFT
elif dx > alignment_threshold:
    x_direction = RIGHT
else:
    x_direction = ALIGNED
```

---

## 7.2 Y direction

Use one consistent convention.

For example:

```text
y_direction:
    behind
    aligned_y
    front
```

Possible rule:

```python
if dy < -alignment_threshold:
    y_direction = BEHIND
elif dy > alignment_threshold:
    y_direction = FRONT
else:
    y_direction = ALIGNED
```

The convention must match the environment coordinate system.

---

## 7.3 X separation

```text
x_separation:
    none
    overlap
    very_near
    near
    far
    very_far
```

---

## 7.4 Y separation

```text
y_separation:
    none
    overlap
    very_near
    near
    far
    very_far
```

---

## 7.5 Room relation

```text
room_relation:
    none
    same_room
    adjacent_room
    nonadjacent_room
```

This helps the model distinguish:

- a large within-room separation;
- a cross-room separation;
- objects separated by multiple room-level transitions.

---

# 8. One factorized edge instead of two physical edges

Conceptually, the relation contains separate X and Y facts:

```text
A --X relation--> B
A --Y relation--> B
```

For the GATv2 implementation, prefer one directed edge:

```text
A -> B:
    x_direction
    x_separation
    y_direction
    y_separation
    room_relation
```

This preserves the fact that the X and Y relations refer to the same source-target pair.

Two separate physical edges may lose this coupling after aggregation.

Therefore:

```text
Conceptual representation:
    two axis relations

Recommended implementation:
    one directed edge with factorized categorical attributes
```

A true multigraph with separate X/Y edges remains possible, but it is better suited to an R-GCN or a custom relation-aware convolution.

---

# 9. Defining separation categories

The graph input remains categorical, but the preprocessing pipeline needs a rule for assigning categories.

Several alternatives are possible.

---

## 9.1 Scene-normalized separation

For cross-room object pairs:

\[
\hat d_x =
\frac{|x_j-x_i|}{\text{scene width}}
\]

\[
\hat d_y =
\frac{|y_j-y_i|}{\text{scene height}}
\]

Then quantize the normalized values.

Example hand-designed thresholds:

```text
overlap:      0.00–0.03
very_near:    0.03–0.15
near:         0.15–0.30
far:          0.30–0.60
very_far:     above 0.60
```

The model sees only the category ID.

---

## 9.2 Room-normalized separation

For objects in the same room:

\[
\hat d_x^{room}
=
\frac{|x_j-x_i|}{\text{room width}}
\]

\[
\hat d_y^{room}
=
\frac{|y_j-y_i|}{\text{room height}}
\]

This retains more meaningful local resolution.

Recommended policy:

```text
same-room edge:
    normalize by room size

cross-room edge:
    normalize by scene size
```

---

## 9.3 Training-set quantile bins

Instead of manually setting thresholds, derive bins from the training set only.

Example:

```text
very_near:  0–25th percentile
near:       25th–50th percentile
far:        50th–75th percentile
very_far:   75th–100th percentile
```

Advantages:

- balanced categorical classes;
- fewer unused categories;
- less dependence on scene scale.

Do not calculate quantiles from validation or test scenes.

---

## 9.4 Strict ordinal rank gaps

For a stronger non-metric experiment, remove magnitude-based bins.

Sort objects by X and Y and define:

```text
x_rank_gap:
    adjacent
    small_gap
    medium_gap
    large_gap

y_rank_gap:
    adjacent
    small_gap
    medium_gap
    large_gap
```

This representation keeps only ordering and ordinal separation.

It is:

- translation invariant;
- scale invariant;
- more strictly non-metric;
- potentially less accurate for fine 36-bin orientation.

Recommended experimental sequence:

1. direction only;
2. direction + normalized qualitative gaps;
3. direction + rank gaps.

---

# 10. Complete categorical edge schema

Each edge stores integer category IDs.

```python
edge_fields = {
    "relation_type": ...,
    "x_direction": ...,
    "x_gap": ...,
    "y_direction": ...,
    "y_gap": ...,
    "room_relation": ...,
}
```

Suggested categories follow.

---

## 10.1 Relation type

```text
SELF
ACTIVE_STAR
GOAL_STAR
OBJECT_INSIDE_ROOM
ROOM_CONTAINS_OBJECT
ROOM_TO_ROOM
SCENE_CONTAINS_ROOM
ROOM_INSIDE_SCENE
```

---

## 10.2 X direction

```text
NONE
LEFT
ALIGNED
RIGHT
```

---

## 10.3 X gap

```text
NONE
OVERLAP
VERY_NEAR
NEAR
FAR
VERY_FAR
```

---

## 10.4 Y direction

```text
NONE
BEHIND
ALIGNED
FRONT
```

---

## 10.5 Y gap

```text
NONE
OVERLAP
VERY_NEAR
NEAR
FAR
VERY_FAR
```

---

## 10.6 Room relation

```text
NONE
SAME_ROOM
ADJACENT_ROOM
NONADJACENT_ROOM
```

Structural edges use `NONE` where a spatial field is not applicable.

---

# 11. Edge encoder

Do not create one giant one-hot category for every possible combination.

Use separate learnable embeddings for each field:

```python
edge_embedding = torch.cat(
    [
        relation_type_embedding[relation_type],
        x_direction_embedding[x_direction],
        x_gap_embedding[x_gap],
        y_direction_embedding[y_direction],
        y_gap_embedding[y_gap],
        room_relation_embedding[room_relation],
    ],
    dim=-1,
)
```

Suggested dimensions:

```text
relation type:  8
x direction:    4
x gap:          6
y direction:    4
y gap:          6
room relation:  4
-----------------
total:         32
```

Therefore:

```python
edge_dim = 32
```

The saved graph should store integer category fields rather than precomputed edge vectors.

Example:

```text
edge_fields.shape = [num_edges, 6]
```

The model turns these category IDs into embeddings.

---

# 12. Recommended encoder

Use a:

> **Hierarchical Edge-Aware GATv2 Encoder**

The encoder consists of:

1. type-specific node initialization;
2. factorized categorical edge encoding;
3. 4 edge-aware GATv2 layers;
4. residual connections;
5. LayerNorm;
6. task-conditioned readout;
7. a 128-dimensional graph representation;
8. a 36-bin orientation head.

---

# 13. Node initialization

## 13.1 Object encoder

```text
CLIP embedding + active + goal
                |
                v
           Object MLP
                |
                v
              128-d
```

Suggested MLP:

```python
object_encoder = nn.Sequential(
    nn.Linear(clip_dim + 2, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
)
```

---

## 13.2 Room encoder

Example raw room vector:

```text
is_room
is_current_room
is_goal_room
```

Suggested encoder:

```python
room_encoder = nn.Sequential(
    nn.Linear(room_feature_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 128),
)
```

---

## 13.3 Scene encoder

Use a learned scene-node embedding:

```python
scene_embedding = nn.Parameter(torch.randn(128))
```

It represents the node type, not a particular scene identity.

---

# 14. Message-passing backbone

Recommended configuration:

```text
hidden dimension: 128
number of layers: 4
attention heads: 4
concat heads: false
edge dimension: 32
dropout: 0.1
residual: yes
LayerNorm: yes
```

Architecture:

```text
128
 |
 v
Edge-aware GATv2
Residual + LayerNorm
 |
 v
Edge-aware GATv2
Residual + LayerNorm
 |
 v
Edge-aware GATv2
Residual + LayerNorm
 |
 v
Edge-aware GATv2
Residual + LayerNorm
 |
 v
128
```

Conceptual layer:

```python
message = conv(
    h,
    edge_index,
    edge_attr,
)

h = norm(
    h + dropout(message)
)
```

Four layers are appropriate because a hierarchical path may be:

```text
active object
    -> current room
    -> goal room
    -> goal object
```

A two-layer GNN may not propagate information across the full path.

---

# 15. Task-conditioned graph readout

Do not use only:

```text
mean_pool(all nodes)
```

Mean pooling can dilute the active and goal nodes among all object, room, and scene nodes.

Explicitly extract:

```text
h_active
h_goal
h_current_room
h_goal_room
h_global
```

Also include pairwise task interactions:

```text
h_goal - h_active
h_goal * h_active
```

Recommended readout:

\[
z =
[
h_a,
h_g,
h_g-h_a,
h_g\odot h_a,
h_{r_a},
h_{r_g},
h_{global}
]
\]

where:

- \(h_a\) is the active-object embedding;
- \(h_g\) is the goal-object embedding;
- \(h_{r_a}\) is the active-room embedding;
- \(h_{r_g}\) is the goal-room embedding;
- \(h_{global}\) is an attention-pooled or mean-pooled global graph embedding.

With a 128-dimensional hidden state:

```text
7 × 128 = 896 dimensions
```

Readout MLP:

```text
896 -> 256 -> 128
```

The final 128-dimensional vector can be passed to the existing GIROL orientation head.
---

# 16. Encoder flow

```text
Object raw features
    CLIP + active + goal
             |
             v
        Object MLP

Room raw features
    current-room + goal-room
             |
             v
         Room MLP

Scene node
    learned type embedding

Categorical edge fields
             |
             v
       Edge embeddings

All nodes + all edge embeddings
             |
             v
    4 × edge-aware GATv2
             |
             v
Extract:
    active
    goal
    active room
    goal room
    global graph
             |
             v
Task-conditioned readout MLP
             |
             v
      128-d graph embedding
             |
             v
      36-bin orientation head
```

---

# 18. PyTorch Geometric-style skeleton

```python
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, global_mean_pool


class EdgeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.relation_emb = nn.Embedding(NUM_RELATION_TYPES, 8)
        self.x_dir_emb = nn.Embedding(NUM_X_DIRECTIONS, 4)
        self.x_gap_emb = nn.Embedding(NUM_X_GAPS, 6)
        self.y_dir_emb = nn.Embedding(NUM_Y_DIRECTIONS, 4)
        self.y_gap_emb = nn.Embedding(NUM_Y_GAPS, 6)
        self.room_rel_emb = nn.Embedding(NUM_ROOM_RELATIONS, 4)

    def forward(self, edge_fields: torch.Tensor) -> torch.Tensor:
        if edge_fields.ndim != 2 or edge_fields.shape[1] != 6:
            raise ValueError(
                "edge_fields must have shape [num_edges, 6]"
            )

        return torch.cat(
            [
                self.relation_emb(edge_fields[:, 0]),
                self.x_dir_emb(edge_fields[:, 1]),
                self.x_gap_emb(edge_fields[:, 2]),
                self.y_dir_emb(edge_fields[:, 3]),
                self.y_gap_emb(edge_fields[:, 4]),
                self.room_rel_emb(edge_fields[:, 5]),
            ],
            dim=-1,
        )


class ResidualEdgeGATLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.conv = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        message = self.conv(
            h,
            edge_index,
            edge_attr=edge_attr,
        )

        return self.norm(
            h + self.dropout(self.activation(message))
        )


class QualitativeGraphEncoder(nn.Module):
    def __init__(
        self,
        clip_dim: int,
        room_feature_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        self.object_encoder = nn.Sequential(
            nn.Linear(clip_dim + 2, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
        )

        self.room_encoder = nn.Sequential(
            nn.Linear(room_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
        )

        self.scene_embedding = nn.Parameter(
            torch.randn(hidden_dim)
        )

        self.edge_encoder = EdgeEncoder()
        edge_dim = 32

        self.layers = nn.ModuleList(
            [
                ResidualEdgeGATLayer(
                    hidden_dim=hidden_dim,
                    edge_dim=edge_dim,
                    heads=heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        readout_dim = hidden_dim * 7

        self.readout = nn.Sequential(
            nn.Linear(readout_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    def forward(
        self,
        object_raw: torch.Tensor,
        room_raw: torch.Tensor,
        node_type: torch.Tensor,
        object_node_indices: torch.Tensor,
        room_node_indices: torch.Tensor,
        scene_node_indices: torch.Tensor,
        edge_index: torch.Tensor,
        edge_fields: torch.Tensor,
        batch: torch.Tensor,
        active_node_index: torch.Tensor,
        goal_node_index: torch.Tensor,
        active_room_index: torch.Tensor,
        goal_room_index: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = node_type.shape[0]

        h = torch.zeros(
            num_nodes,
            self.hidden_dim,
            dtype=object_raw.dtype,
            device=object_raw.device,
        )

        h[object_node_indices] = self.object_encoder(
            object_raw
        )

        h[room_node_indices] = self.room_encoder(
            room_raw
        )

        h[scene_node_indices] = self.scene_embedding

        edge_attr = self.edge_encoder(edge_fields)

        for layer in self.layers:
            h = layer(
                h=h,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )

        h_active = h[active_node_index]
        h_goal = h[goal_node_index]
        h_active_room = h[active_room_index]
        h_goal_room = h[goal_room_index]

        h_global = global_mean_pool(h, batch)

        graph_repr = torch.cat(
            [
                h_active,
                h_goal,
                h_goal - h_active,
                h_goal * h_active,
                h_active_room,
                h_goal_room,
                h_global,
            ],
            dim=-1,
        )

        return self.readout(graph_repr)


class OrientationModel(nn.Module):
    def __init__(
        self,
        graph_encoder: QualitativeGraphEncoder,
        latent_dim: int = 128,
        num_yaw_bins: int = 36,
    ) -> None:
        super().__init__()

        self.graph_encoder = graph_encoder

        self.orientation_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_yaw_bins),
        )

    def forward(self, **graph_inputs) -> torch.Tensor:
        graph_embedding = self.graph_encoder(
            **graph_inputs
        )

        return self.orientation_head(graph_embedding)
```

---

# 19. Graph construction pseudocode

```python
def build_non_metric_graph(
    objects,
    rooms,
    active_object_id,
    goal_object_id,
    train_quantiles,
):
    nodes = []
    edges = []
    edge_fields = []

    scene_node = add_scene_node(nodes)

    room_nodes = {}
    for room in rooms:
        room_nodes[room.id] = add_room_node(
            nodes=nodes,
            is_current_room=room.contains(active_object_id),
            is_goal_room=room.contains(goal_object_id),
        )

        add_bidirectional_structural_edge(
            edges=edges,
            edge_fields=edge_fields,
            source=scene_node,
            target=room_nodes[room.id],
            forward_relation=SCENE_CONTAINS_ROOM,
            reverse_relation=ROOM_INSIDE_SCENE,
            room_global_x_zone=quantize_room_x_zone(room),
            room_global_y_zone=quantize_room_y_zone(room),
        )

    object_nodes = {}
    for obj in objects:
        object_nodes[obj.id] = add_object_node(
            nodes=nodes,
            clip_embedding=obj.clip_embedding,
            active=obj.id == active_object_id,
            is_goal=obj.id == goal_object_id,
        )

        room_node = room_nodes[obj.room_id]

        add_bidirectional_structural_edge(
            edges=edges,
            edge_fields=edge_fields,
            source=object_nodes[obj.id],
            target=room_node,
            forward_relation=OBJECT_INSIDE_ROOM,
            reverse_relation=ROOM_CONTAINS_OBJECT,
        )

    for source_room in rooms:
        for target_room in rooms:
            if source_room.id == target_room.id:
                continue

            add_room_relation_edge(
                edges=edges,
                edge_fields=edge_fields,
                source=room_nodes[source_room.id],
                target=room_nodes[target_room.id],
                relation=qualitative_room_relation(
                    source_room,
                    target_room,
                ),
            )

    active_node = object_nodes[active_object_id]
    goal_node = object_nodes[goal_object_id]

    relevant_object_ids = [
        obj.id for obj in objects
    ]

    for target_id in relevant_object_ids:
        target_node = object_nodes[target_id]

        add_qualitative_object_edge(
            source_object_id=active_object_id,
            target_object_id=target_id,
            source_node=active_node,
            target_node=target_node,
            relation_type=ACTIVE_STAR,
            train_quantiles=train_quantiles,
            edges=edges,
            edge_fields=edge_fields,
        )

        add_qualitative_object_edge(
            source_object_id=goal_object_id,
            target_object_id=target_id,
            source_node=goal_node,
            target_node=target_node,
            relation_type=GOAL_STAR,
            train_quantiles=train_quantiles,
            edges=edges,
            edge_fields=edge_fields,
        )

    add_self_loops(
        nodes=nodes,
        edges=edges,
        edge_fields=edge_fields,
        relation_type=SELF,
    )

    return nodes, edges, edge_fields
```