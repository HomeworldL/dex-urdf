# `migrate_hand_urdf_to_mjcf.py` 执行树状图（中文）

本文档按 `tools/migrate_hand_urdf_to_mjcf.py` 当前代码顺序，整理函数的主执行路径、子流程调用关系，以及每一组函数的用途说明。目的是让你在读脚本时，可以直接对照“入口 -> 中间步骤 -> 输出文件”的结构。

## 1. 主入口执行树

```text
__main__
└── main(_parse_args())
    ├── _parse_args()
    │   └── 返回 Config
    └── main(cfg)
        ├── 定位 repo_root / hand_dir
        ├── _sides_to_process(cfg, hand_dir)
        │   ├── 若 side 为 left/right
        │   │   └── 直接返回
        │   └── 若 side 为 all
        │       └── _pick_side_source_urdf() x 2
        └── for side in sides:
            └── _process_side(cfg, repo_root, hand_dir, side)
                ├── 选择源 URDF
                │   └── _pick_source_urdf()
                ├── _output_stem()
                ├── _clean_generated_outputs()
                │   ├── _asset_refs_from_xml() for old xml/urdf
                │   └── 删除旧 mesh / xml / urdf
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
                │   │   ├── .obj 直出路径
                │   │   │   └── _export_plain_obj()
                │   │   │       ├── _load_meshes()
                │   │   │       ├── _scaled_mesh()
                │   │   │       └── _write_obj_with_unique_assets()
                │   │   └── glb/scene 拆分路径
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
                │   ├── link/collision 遍历
                │   └── _collision_bounds()
                │       ├── _parse_xyz()
                │       └── _parse_mesh_scale()
                ├── 写出 *_for_mjcf.urdf
                ├── 若 cfg.try_compile_mjcf:
                │   └── _try_compile_to_mjcf()
                │       ├── _make_visual_preserving_urdf()
                │       ├── mujoco.MjModel.from_xml_path()
                │       ├── mujoco.mj_saveLastXML()
                │       ├── _restore_mjcf_mesh_paths()
                │       └── _patch_visual_materials()
                │           ├── _obj_material()
                │           └── _parse_mtl()
                ├── 若 cfg.patch_mjcf:
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

## 2. 按阶段拆开的子流程

### 2.1 参数与输入选择

```text
_parse_args
└── Config

main
└── _sides_to_process
    └── _pick_side_source_urdf

_process_side
└── _pick_source_urdf
```

说明：

- `Config` 是整个脚本的运行配置。
- `main()` 只负责确定处理哪些 side，然后逐个交给 `_process_side()`。
- `_pick_source_urdf()` 是单 side 真正使用哪个 URDF 的最终选择函数。

### 2.2 mesh 读取、缩放、导出

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

说明：

- visual 和 collision 现在都先读 mesh，再把 URDF 里的 `scale` 烘进导出的 OBJ 顶点。
- visual 路径会尽量保留颜色材质；collision 路径只关心可编译和稳定的几何。
- `_is_valid_visual_mesh()` / `_is_valid_collision_mesh()` 都用于剔除退化几何，但 collision 这边的过滤更关键，因为它直接影响 MuJoCo / QHull。

### 2.3 URDF 规范化

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

说明：

- `_ensure_mujoco_compiler()` 会统一写入 `<mujoco><compiler .../></mujoco>`。
- `_normalize_hand_root()` 的目标是让 prepared URDF 顶层有统一的 `hand_root` 和 `hand_root_joint`。
- `_ensure_missing_inertials()` 只给“有 collision 但没 inertial”的 link 补一个保守的惯量。

### 2.4 MJCF 编译与材质修补

```text
_try_compile_to_mjcf
├── _make_visual_preserving_urdf
├── mujoco compile
├── _restore_mjcf_mesh_paths
└── _patch_visual_materials
    ├── _obj_material
    └── _parse_mtl
```

说明：

- `_make_visual_preserving_urdf()` 会临时生成一个 `discardvisual=false` 的编译版本，避免 MuJoCo 丢 visual。
- `_restore_mjcf_mesh_paths()` 用来把 MuJoCo 保存 XML 后可能被改乱的 mesh 路径修回来。
- `_patch_visual_materials()` 会读取 visual OBJ 对应的 MTL，把 `Kd` 或 texture 映射成 MJCF 的 `<material>` / `<texture>`。

### 2.5 MJCF 二次补丁

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

说明：

- 这一层是在 MuJoCo 已经把 URDF 编译成 XML 之后，再按本仓库习惯做整理。
- 主要包括：全局设置、光照、关节参数、visual/collision 分组、父子 body contact exclude、mimic 约束、position actuator。

### 2.6 collision-only XML

```text
_write_collision_only_mjcf
└── 删除所有 visual geom
    ├── 保留 collision geom
    ├── 清空 material
    ├── 设置半透明橙色 rgba
    └── 删掉不再引用的 material / mesh
```

说明：

- 该输出是为了单独检查 collision 体是否正确。
- visual geom 会被完全移除，只保留 collision mesh / primitive。

## 3. 文件输出关系

对单个 `*_glb.urdf`，脚本最终会生成：

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

说明：

- `<name>_original.urdf`：原始 URDF 经过 `ElementTree` 重新格式化后的副本，方便对照。
- `<name>_for_mjcf.urdf`：真正送去 MuJoCo 编译的 prepared URDF。
- `<name>.xml`：编译并 patch 后的主 MJCF。
- `<name>_collision.xml`：只显示碰撞体的 MJCF。

## 4. 关键函数速查

### 4.1 入口与编排

- `main`：全局入口。
- `_process_side`：单个 side 的完整迁移流程。
- `_clean_generated_outputs`：删除旧输出，避免历史文件残留干扰本次结果。

### 4.2 visual 导出

- `_rewrite_visual_meshes`：把 URDF 中的 visual mesh 改写到导出的 OBJ。
- `_export_visual_objs`：GLB 按内部几何拆 OBJ；OBJ 则直接按一个输出处理。
- `_patch_visual_materials`：把 MTL 里的颜色/贴图转成 MJCF material。

### 4.3 collision 导出

- `_rewrite_collision_meshes`：把 URDF collision mesh 改写为导出的 OBJ。
- `_split_obj_by_object`：把 collision mesh 拆成多个连通块，并剔除退化块。
- `_write_collision_only_mjcf`：单独输出只看 collision 的 XML。

### 4.4 URDF 结构修正

- `_normalize_hand_root`：统一根 link/joint 为 `hand_root` / `hand_root_joint`。
- `_ensure_missing_inertials`：基于 collision 包围盒补惯量。
- `_ensure_mujoco_compiler`：统一插入 MuJoCo 编译器配置。

### 4.5 MJCF 后处理

- `patch_mjcf`：MJCF 总补丁入口。
- `patch_contact_excludes`：自动为父子 body 添加 exclude。
- `patch_actuators`：为 active joint 添加 position actuator。
- `patch_mimic_equalities`：把 URDF mimic 关系变成 MJCF equality。

## 5. 当前脚本的实际处理顺序

如果只看最核心的顺序，可以压缩成下面这棵树：

```text
main
└── _process_side
    ├── 选 URDF
    ├── 清理旧输出
    ├── 保存 original URDF
    ├── 规范化 URDF 根结构
    ├── 导出 visual meshes
    ├── 导出 collision meshes
    ├── 补 inertial
    ├── 写 prepared URDF
    ├── 编译成 MJCF
    ├── 修 mesh/material
    ├── patch MJCF
    └── 写 collision-only MJCF
```

这就是当前脚本最重要的主干。
