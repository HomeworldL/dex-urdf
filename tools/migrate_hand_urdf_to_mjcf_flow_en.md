# `migrate_hand_urdf_to_mjcf.py` Execution Tree (English)

This document summarizes the current function order and call structure of `tools/migrate_hand_urdf_to_mjcf.py`. The goal is to make the script easy to read as a pipeline: entry point -> transformation stages -> generated outputs.

## 1. Top-Level Execution Tree

```text
__main__
└── main(_parse_args())
    ├── _parse_args()
    │   └── returns Config
    └── main(cfg)
        ├── resolve repo_root / hand_dir
        ├── _sides_to_process(cfg, hand_dir)
        │   ├── if side is left/right
        │   │   └── return directly
        │   └── if side == all
        │       └── _pick_side_source_urdf() x 2
        └── for side in sides:
            └── _process_side(cfg, repo_root, hand_dir, side)
                ├── choose source URDF
                │   └── _pick_source_urdf()
                ├── _output_stem()
                ├── _clean_generated_outputs()
                │   ├── _asset_refs_from_xml() for old xml/urdf
                │   └── remove old mesh / xml / urdf
                ├── _write_elementtree_copy()
                ├── ET.parse(src_urdf)
                ├── _ensure_mujoco_compiler()
                ├── _normalize_hand_root()
                │   ├── _root_links()
                │   ├── _link_has_geometry()
                │   └── _rename_link_references()
                ├── _rewrite_visual_meshes()
                │   ├── _parse_mesh_scale()
                │   ├── _export_visual_objs()
                │   │   ├── direct .obj path
                │   │   │   └── _export_plain_obj()
                │   │   │       ├── _load_meshes()
                │   │   │       ├── _scaled_mesh()
                │   │   │       └── _write_obj_with_unique_assets()
                │   │   └── glb/scene split path
                │   │       ├── _load_meshes()
                │   │       ├── _scaled_mesh()
                │   │       ├── _is_valid_visual_mesh()
                │   │       └── _write_obj_with_unique_assets()
                │   └── _check_unique_output()
                ├── _rewrite_collision_meshes()
                │   ├── _parse_mesh_scale()
                │   └── _split_obj_by_object()
                │       ├── _load_meshes()
                │       ├── _scaled_mesh()
                │       ├── _is_valid_collision_mesh()
                │       └── _write_obj_with_unique_assets()
                ├── _ensure_missing_inertials()
                │   ├── iterate link/collision
                │   └── _collision_bounds()
                │       ├── _parse_xyz()
                │       └── _parse_mesh_scale()
                ├── write *_for_mjcf.urdf
                ├── if cfg.try_compile_mjcf:
                │   └── _try_compile_to_mjcf()
                │       ├── _make_visual_preserving_urdf()
                │       ├── mujoco.MjModel.from_xml_path()
                │       ├── mujoco.mj_saveLastXML()
                │       ├── _restore_mjcf_mesh_paths()
                │       └── _patch_visual_materials()
                │           ├── _obj_material()
                │           └── _parse_mtl()
                ├── if cfg.patch_mjcf:
                │   ├── patch_mjcf()
                │   │   ├── _mimic_specs()
                │   │   ├── _active_joints()
                │   │   │   └── _mimic_specs()
                │   │   ├── patch_global_settings()
                │   │   │   ├── _find_child()
                │   │   │   └── _replace_child()
                │   │   ├── patch_worldbody()
                │   │   │   └── _find_child()
                │   │   ├── patch_joint_params()
                │   │   ├── patch_geom_groups()
                │   │   ├── patch_contact_excludes()
                │   │   │   ├── _body_parent_pairs()
                │   │   │   └── _replace_child()
                │   │   ├── patch_mimic_equalities()
                │   │   │   └── _replace_child()
                │   │   ├── patch_actuators()
                │   │   │   ├── _joint_limits()
                │   │   │   └── _replace_child()
                │   │   └── _order_top_level()
                │   └── _try_compile_xml()
                └── _write_collision_only_mjcf()
                    └── _try_compile_xml()
```

## 2. Stage-by-Stage Subtrees

### 2.1 Argument parsing and source selection

```text
_parse_args
└── Config

main
└── _sides_to_process
    └── _pick_side_source_urdf

_process_side
└── _pick_source_urdf
```

Notes:

- `Config` is the runtime configuration object.
- `main()` only decides which sides should be processed.
- `_pick_source_urdf()` is the final selector for the actual source URDF of one side.

### 2.2 Mesh loading, scaling, and export

```text
_rewrite_visual_meshes
├── _parse_mesh_scale
├── _export_visual_objs
│   ├── _export_plain_obj
│   │   ├── _load_meshes
│   │   ├── _scaled_mesh
│   │   └── _write_obj_with_unique_assets
│   ├── _load_meshes
│   ├── _scaled_mesh
│   ├── _is_valid_visual_mesh
│   └── _write_obj_with_unique_assets
└── _check_unique_output

_rewrite_collision_meshes
├── _parse_mesh_scale
└── _split_obj_by_object
    ├── _load_meshes
    ├── _scaled_mesh
    ├── _is_valid_collision_mesh
    └── _write_obj_with_unique_assets
```

Notes:

- Both visual and collision paths now bake URDF mesh scale into exported OBJ vertices.
- The visual path tries to preserve material/color information.
- The collision path focuses on stable, compilable geometry for MuJoCo.

### 2.3 URDF normalization

```text
_process_side
├── _ensure_mujoco_compiler
├── _normalize_hand_root
│   ├── _root_links
│   ├── _link_has_geometry
│   └── _rename_link_references
└── _ensure_missing_inertials
    └── _collision_bounds
        ├── _parse_xyz
        └── _parse_mesh_scale
```

Notes:

- `_ensure_mujoco_compiler()` inserts a normalized `<mujoco><compiler .../></mujoco>` block.
- `_normalize_hand_root()` ensures a consistent `hand_root` and `hand_root_joint` in the prepared URDF.
- `_ensure_missing_inertials()` adds conservative inertials only when a physical link has collision geometry but no inertial block.

### 2.4 MJCF compilation and material recovery

```text
_try_compile_to_mjcf
├── _make_visual_preserving_urdf
├── mujoco compile
├── _restore_mjcf_mesh_paths
└── _patch_visual_materials
    ├── _obj_material
    └── _parse_mtl
```

Notes:

- `_make_visual_preserving_urdf()` creates a temporary URDF with `discardvisual=false`.
- `_restore_mjcf_mesh_paths()` fixes mesh file paths after MuJoCo saves the XML.
- `_patch_visual_materials()` reads the exported OBJ/MTL files and maps `Kd` / textures into MJCF `<material>` and `<texture>` assets.

### 2.5 MJCF post-patching

```text
patch_mjcf
├── _mimic_specs
├── _active_joints
│   └── _mimic_specs
├── patch_global_settings
│   ├── _find_child
│   └── _replace_child
├── patch_worldbody
│   └── _find_child
├── patch_joint_params
├── patch_geom_groups
├── patch_contact_excludes
│   ├── _body_parent_pairs
│   └── _replace_child
├── patch_mimic_equalities
│   └── _replace_child
├── patch_actuators
│   ├── _joint_limits
│   └── _replace_child
└── _order_top_level
```

Notes:

- This stage runs after MuJoCo has already converted the prepared URDF into XML.
- It applies project-specific MJCF conventions: global settings, lighting, joint params, visual/collision groups, parent-child contact excludes, mimic equalities, and position actuators.

### 2.6 Collision-only XML

```text
_write_collision_only_mjcf
└── remove all visual geoms
    ├── keep collision geoms
    ├── clear material
    ├── assign translucent orange rgba
    └── remove unreferenced material / mesh assets
```

Notes:

- This output is meant for collision inspection only.
- Visual geoms are removed completely.

## 3. Output File Structure

For one `*_glb.urdf`, the script produces:

```text
robots_mjcf/hands/<hand_name>/
├── <name>_original.urdf
├── <name>_for_mjcf.urdf
├── <name>.xml
├── <name>_collision.xml
└── meshes/
    ├── visual/
    │   ├── *.obj
    │   └── *.mtl
    └── collision/
        └── *.obj
```

Notes:

- `<name>_original.urdf`: ElementTree-formatted copy of the source URDF for diffing.
- `<name>_for_mjcf.urdf`: prepared URDF actually used for MuJoCo compilation.
- `<name>.xml`: main MJCF after compilation and patching.
- `<name>_collision.xml`: collision-only MJCF.

## 4. Quick Function Index

### 4.1 Entry and orchestration

- `main`: top-level entry.
- `_process_side`: full migration pipeline for one side.
- `_clean_generated_outputs`: removes stale outputs from previous runs.

### 4.2 Visual export

- `_rewrite_visual_meshes`: rewrites visual mesh references to exported OBJ assets.
- `_export_visual_objs`: splits GLB scene geometry into OBJ files or exports OBJ directly.
- `_patch_visual_materials`: converts MTL color/texture data into MJCF material assets.

### 4.3 Collision export

- `_rewrite_collision_meshes`: rewrites collision mesh references to exported OBJ assets.
- `_split_obj_by_object`: splits collision meshes into connected components and drops degenerate pieces.
- `_write_collision_only_mjcf`: emits the collision-only XML view.

### 4.4 URDF structural fixes

- `_normalize_hand_root`: normalizes the root link/joint to `hand_root` / `hand_root_joint`.
- `_ensure_missing_inertials`: adds conservative inertials from collision bounds.
- `_ensure_mujoco_compiler`: inserts a normalized MuJoCo compiler block.

### 4.5 MJCF post-processing

- `patch_mjcf`: main MJCF post-patch entry.
- `patch_contact_excludes`: auto-generates excludes for parent-child bodies.
- `patch_actuators`: creates position actuators for active joints.
- `patch_mimic_equalities`: converts URDF mimic relations into MJCF equalities.

## 5. Minimal Core Pipeline

If you only want the shortest possible summary, the current script behaves like this:

```text
main
└── _process_side
    ├── choose URDF
    ├── clean old outputs
    ├── save original URDF
    ├── normalize URDF root structure
    ├── export visual meshes
    ├── export collision meshes
    ├── add inertials
    ├── write prepared URDF
    ├── compile MJCF
    ├── restore mesh/material information
    ├── patch MJCF
    └── write collision-only MJCF
```

That is the main backbone of the current implementation.
