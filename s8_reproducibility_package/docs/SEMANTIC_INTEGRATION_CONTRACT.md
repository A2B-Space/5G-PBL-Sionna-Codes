# Proposed semantic-to-physical-layer integration contract

This document defines a future interface. The current S6/S7 data does NOT
contain camera images, masks, bounding boxes, semantic classes or synchronized
cross-modal labels.

## Existing physical-layer keys

The physical-layer tables carry `scene_id`, `trajectory_id`, `condition_id`,
`snapshot`, and `sample_id`. `snapshot_metadata.csv` also contains receiver XYZ
coordinates and the LOS/blockage/channel/beam descriptors. Use `sample_id` as
the primary join key after verifying it is unique in each table at its declared
grain.

## Required semantic-side fields before joining

- `scene_id`
- `trajectory_id`
- `snapshot` or an explicit timestamp-to-snapshot mapping
- `camera_timestamp` with declared clock/time base
- camera pose and coordinate frame
- camera calibration and field of view
- object/instance IDs and semantic labels
- object geometry or 3D location if a physical object-to-ray interaction link
  is intended

## Join rules

1. Establish a common coordinate frame and document every transform.
2. Establish synchronization tolerance and timestamp provenance; do not join
   by row order alone.
3. Preserve one-to-many relationships: one snapshot can have multiple paths,
   multiple interactions, 52 beam measurements and multiple semantic objects.
4. Keep `sample_id` as the sample-level key and use `(sample_id, path_id)` or
   `(sample_id, beam_id)` for child tables.
5. Do not infer that a ray interaction object ID is a camera object ID. A
   geometry/object correspondence table must be explicitly constructed and
   validated.
6. Keep train/validation/test splits grouped by independent trajectory or
   scene, not by adjacent snapshots from the same trajectory.

## Suggested future correspondence table

`sample_id, semantic_object_id, scene_object_id, correspondence_method,
confidence, coordinate_transform_id, timestamp_delta_ms`

This table is a proposed integration artifact, not an existing S6 output.
