# Room Setup Replica V1 Replay

This is an isolated, replay-only V1 comparator. It does not modify anything
under `Assets/RoomSetupReplica`, and it does not instantiate ScanCover's
production reconstruction route.

Use Unity menu `Room Setup Replica V1 Replay > 重建独立测试场景`. The generated
scene reads the exact same Quest Stage tape as V2:

`Application.persistentDataPath/RoomSetupReplicaV2/last-session.rsv2`

Controls:

- A: clear and replay the shared V2 tape through the V1 comparator.
- B: clear the comparator volume and mesh.
- X: show or hide the comparator mesh.

The controller actions share the same 250 ms debounce as V2. The right trigger
is intentionally inactive in this scene so the comparator cannot accidentally
overwrite the common input tape.

The common replay harness, fusion, truncation, scheduling, Stage alignment,
export, and metrics are identical to V2. The controlled algorithm difference
is the V1 meshing rule: the 10 cm Marching Cubes lattice directly takes every
second voxel from the 5 cm source field, including the original half-voxel
sample-center offset. It does not use V2's weighted eight-neighbour derived
field.

Outputs are kept separate under
`Application.persistentDataPath/RoomSetupReplicaV1Replay/`:

- `last-replay-mesh.ply`
- `last-replay-receipt.json`

Both V1 and V2 receipts report deterministic mesh identity, replay timing,
connected components, planar residual, and direct-depth support. Direct-depth
support is measured by reprojecting every final triangle against every stereo
packet. A triangle is supported when its centroid agrees with either eye's
depth within 5 cm, or at least two vertices agree within the same packet.
TSDF weights are not accepted as a substitute for this measurement.

This comparator establishes a same-input A/B experiment. It does not claim
bit-identical behavior to Meta Room Setup and still requires Quest device
validation.
