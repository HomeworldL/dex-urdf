from __future__ import annotations

import copy
import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import trimesh
from trimesh.exchange.obj import export_obj


#############
# Configuration
#############
@dataclass
class Config:
    hand_name: str
    side: str = "all"  # all | right | left
    output_root: str = "robots_mjcf/hands"
    clean_output: bool = True
    try_compile_mjcf: bool = True
    patch_mjcf: bool = True
    kp: float = 1.0


#############
# MJCF patching
#############
ACTIVE_JOINT_ATTRS = {
    "damping": "0.1",
    "armature": "0.0002",
    "frictionloss": "0.0",
}
MIMIC_JOINT_ATTRS = {
    "damping": "0.0",
    "armature": "0.0",
    "frictionloss": "0.0",
}
MIMIC_EQUALITY_ATTRS = {
    "solref": "0.002 1",
    "solimp": "0.9 0.99 0.001",
}


def _joint_limits(urdf_root: ET.Element) -> Dict[str, Tuple[str, str]]:
    limits: Dict[str, Tuple[str, str]] = {}
    for joint in urdf_root.findall(".//joint"):
        if joint.attrib.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        lower = limit.attrib.get("lower")
        upper = limit.attrib.get("upper")
        if lower is not None and upper is not None:
            limits[joint.attrib["name"]] = (lower, upper)
    return limits


def _mimic_specs(urdf_root: ET.Element) -> List[Tuple[str, str, str, str]]:
    specs = []
    for joint in urdf_root.findall(".//joint"):
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        specs.append(
            (
                joint.attrib["name"],
                mimic.attrib["joint"],
                mimic.attrib.get("multiplier", "1"),
                mimic.attrib.get("offset", "0"),
            )
        )
    return specs


def _active_joints(urdf_root: ET.Element) -> List[str]:
    mimic_joints = {joint for joint, *_ in _mimic_specs(urdf_root)}
    active = []
    for joint in urdf_root.findall(".//joint"):
        name = joint.attrib["name"]
        if joint.attrib.get("type") == "fixed" or name in mimic_joints:
            continue
        active.append(name)
    return active


def _replace_child(root: ET.Element, tag: str, child: ET.Element) -> None:
    for existing in list(root):
        if existing.tag == tag:
            root.remove(existing)
    root.append(child)


def _order_top_level(mjcf_root: ET.Element) -> None:
    order = {
        "compiler": 0,
        "option": 1,
        "visual": 2,
        "asset": 3,
        "worldbody": 4,
        "contact": 5,
        "equality": 6,
        "actuator": 7,
    }
    mjcf_root[:] = sorted(list(mjcf_root), key=lambda child: order.get(child.tag, 100))


def _find_child(root: ET.Element, tag: str) -> ET.Element:
    child = root.find(tag)
    if child is None:
        child = ET.SubElement(root, tag)
    return child


def _body_parent_pairs(root: ET.Element) -> Iterable[Tuple[str, str]]:
    def walk(parent: ET.Element) -> Iterable[Tuple[str, str]]:
        parent_name = parent.attrib.get("name")
        if parent_name is not None:
            for child in parent.findall("body"):
                child_name = child.attrib.get("name")
                if child_name is not None:
                    yield parent_name, child_name
                yield from walk(child)

    worldbody = root.find("worldbody")
    if worldbody is None:
        return []
    pairs: List[Tuple[str, str]] = []
    for body in worldbody.findall("body"):
        pairs.extend(walk(body))
    return pairs


def patch_global_settings(mjcf_root: ET.Element) -> None:
    compiler = _find_child(mjcf_root, "compiler")
    compiler.attrib["angle"] = "radian"
    compiler.attrib["autolimits"] = "true"

    option = _find_child(mjcf_root, "option")
    option.attrib["gravity"] = "0 0 0"
    option.attrib["timestep"] = "0.001"

    asset = _find_child(mjcf_root, "asset")
    for texture in list(asset.findall("texture")):
        if texture.attrib.get("type") == "skybox":
            asset.remove(texture)
    ET.SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1="1 1 1",
        rgb2="1 1 1",
        width="32",
        height="512",
    )

    visual = ET.Element("visual")
    ET.SubElement(visual, "global", offwidth="3840", offheight="2160")
    _replace_child(mjcf_root, "visual", visual)


def patch_worldbody(mjcf_root: ET.Element) -> None:
    worldbody = _find_child(mjcf_root, "worldbody")
    for light in list(worldbody.findall("light")):
        worldbody.remove(light)
    worldbody.insert(0, ET.Element("light", pos="0 0 5", dir="0 0 -1", directional="true"))


def patch_joint_params(mjcf_root: ET.Element, active_joints: Iterable[str], mimic_joints: Iterable[str]) -> None:
    active = set(active_joints)
    mimic = set(mimic_joints)
    for joint in mjcf_root.findall(".//joint"):
        name = joint.attrib.get("name")
        if name in active:
            joint.attrib.update(ACTIVE_JOINT_ATTRS)
        elif name in mimic:
            joint.attrib.update(MIMIC_JOINT_ATTRS)


def patch_geom_groups(mjcf_root: ET.Element) -> None:
    for geom in mjcf_root.findall(".//geom"):
        if geom.attrib.get("contype") == "0" and geom.attrib.get("conaffinity") == "0":
            geom.attrib["group"] = "1"
        else:
            geom.attrib["group"] = "3"


def patch_contact_excludes(mjcf_root: ET.Element) -> None:
    contact = ET.Element("contact")
    for body1, body2 in _body_parent_pairs(mjcf_root):
        if body1 == "hand_root" or body2 == "hand_root":
            continue
        ET.SubElement(contact, "exclude", name=f"ex_{body1}_{body2}", body1=body1, body2=body2)
    _replace_child(mjcf_root, "contact", contact)


def patch_mimic_equalities(mjcf_root: ET.Element, mimic_specs: Iterable[Tuple[str, str, str, str]]) -> None:
    equality = ET.Element("equality")
    for mimic_joint, parent_joint, multiplier, offset in mimic_specs:
        ET.SubElement(
            equality,
            "joint",
            name=f"{mimic_joint}_mimic",
            joint1=mimic_joint,
            joint2=parent_joint,
            polycoef=f"{offset} {multiplier} 0 0 0",
            **MIMIC_EQUALITY_ATTRS,
        )
    _replace_child(mjcf_root, "equality", equality)


def patch_actuators(
    mjcf_root: ET.Element,
    active_joints: Iterable[str],
    joint_limits: Dict[str, Tuple[str, str]],
    kp: float,
) -> None:
    actuator = ET.Element("actuator")
    for joint_name in active_joints:
        attrs = {
            "name": f"{joint_name}_actuator",
            "joint": joint_name,
            "kp": f"{kp:g}",
        }
        if joint_name in joint_limits:
            attrs["ctrlrange"] = f"{joint_limits[joint_name][0]} {joint_limits[joint_name][1]}"
        ET.SubElement(actuator, "position", **attrs)
    _replace_child(mjcf_root, "actuator", actuator)


def patch_mjcf(urdf_path: Path, mjcf_path: Path, kp: float) -> None:
    """Apply project MJCF defaults after MuJoCo converts the prepared URDF."""
    urdf_root = ET.parse(urdf_path).getroot()
    mjcf_tree = ET.parse(mjcf_path)
    mjcf_root = mjcf_tree.getroot()

    mimic_specs = _mimic_specs(urdf_root)
    mimic_joints = [joint for joint, *_ in mimic_specs]
    active_joints = _active_joints(urdf_root)

    patch_global_settings(mjcf_root)
    patch_worldbody(mjcf_root)
    patch_joint_params(mjcf_root, active_joints, mimic_joints)
    patch_geom_groups(mjcf_root)
    patch_contact_excludes(mjcf_root)
    patch_mimic_equalities(mjcf_root, mimic_specs)
    patch_actuators(mjcf_root, active_joints, _joint_limits(urdf_root), kp)
    _order_top_level(mjcf_root)

    ET.indent(mjcf_tree, space="  ")
    mjcf_tree.write(mjcf_path, encoding="utf-8")


#############
# Source selection
#############
def _pick_source_urdf(hand_dir: Path, hand_name: str, side: str) -> Path:
    if side not in {"right", "left"}:
        raise ValueError("side must be 'right' or 'left'")

    preferred = [
        hand_dir / f"{hand_name}_{side}_glb.urdf",
        hand_dir / f"{hand_name}_{side}.urdf",
        hand_dir / f"{hand_name}_glb.urdf",
        hand_dir / f"{hand_name}.urdf",
    ]
    for p in preferred:
        if p.exists():
            return p

    candidates = sorted(hand_dir.glob("*.urdf"))
    if not candidates:
        raise FileNotFoundError(f"No urdf found under {hand_dir}")
    return candidates[0]


def _pick_side_source_urdf(hand_dir: Path, hand_name: str, side: str) -> Optional[Path]:
    for p in [
        hand_dir / f"{hand_name}_{side}_glb.urdf",
        hand_dir / f"{hand_name}_{side}.urdf",
    ]:
        if p.exists():
            return p
    return None


def _sides_to_process(cfg: Config, hand_dir: Path) -> List[str]:
    if cfg.side in {"right", "left"}:
        return [cfg.side]
    if cfg.side != "all":
        raise ValueError("side must be 'all', 'right', or 'left'")

    sides = [
        side
        for side in ["right", "left"]
        if _pick_side_source_urdf(hand_dir, cfg.hand_name, side) is not None
    ]
    return sides or ["right"]


#############
# Mesh loading and asset export
#############
def _load_meshes(mesh_path: Path) -> List[trimesh.Trimesh]:
    scene = trimesh.load(str(mesh_path), force="scene", process=False)
    meshes: List[trimesh.Trimesh] = []
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        geom = scene.geometry[geometry_name]
        if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0:
            mesh = geom.copy()
            mesh.apply_transform(transform)
            meshes.append(mesh)
    if meshes:
        return meshes

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) > 0:
        return [mesh]
    return []


def _check_unique_output(seen: Dict[str, str], src_rel: str, output_name: str) -> None:
    previous = seen.get(output_name)
    if previous is not None and previous != src_rel:
        raise RuntimeError(f"Mesh output name conflict: {previous} and {src_rel} both map to {output_name}")
    seen[output_name] = src_rel


def _strip_mtl_texture_maps(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("map_")) + "\n"


def _write_obj_with_unique_assets(
    mesh: trimesh.Trimesh,
    dst_mesh: Path,
    include_materials: bool,
) -> None:
    dst_mesh.parent.mkdir(parents=True, exist_ok=True)
    obj_text, textures = export_obj(
        mesh,
        include_texture=include_materials,
        mtl_name=dst_mesh.with_suffix(".mtl").name,
        return_texture=True,
    )

    material_prefix = dst_mesh.stem
    obj_text = obj_text.replace("usemtl material_", f"usemtl {material_prefix}_material_")

    dst_mesh.write_text(obj_text)

    for name, data in textures.items():
        if name != dst_mesh.with_suffix(".mtl").name:
            continue

        if isinstance(data, str):
            text = data
        else:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue

        if name == dst_mesh.with_suffix(".mtl").name:
            text = text.replace("newmtl material_", f"newmtl {material_prefix}_material_")
            text = _strip_mtl_texture_maps(text)
        (dst_mesh.parent / name).write_text(text)


def _scaled_mesh(mesh: trimesh.Trimesh, scale: Tuple[float, float, float]) -> trimesh.Trimesh:
    scaled = mesh.copy()
    scaled.apply_scale(scale)
    return scaled


def _export_plain_obj(
    src_mesh: Path,
    dst_mesh: Path,
    scale: Tuple[float, float, float],
    include_materials: bool,
) -> Path:
    meshes = _load_meshes(src_mesh)
    if not meshes:
        raise RuntimeError(f"No valid mesh geometry in {src_mesh}")
    merged = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    _write_obj_with_unique_assets(_scaled_mesh(merged, scale), dst_mesh, include_materials=include_materials)
    return dst_mesh


def _is_valid_visual_mesh(mesh: trimesh.Trimesh) -> bool:
    if len(mesh.faces) < 4 or len(mesh.vertices) < 4:
        return False

    vertices = np.unique(np.asarray(mesh.vertices, dtype=float), axis=0)
    if len(vertices) < 4:
        return False

    extents = np.ptp(vertices, axis=0)
    rank_scale = max(float(extents.max()), 1.0)
    rank = np.linalg.matrix_rank(vertices - vertices.mean(axis=0), tol=rank_scale * 1e-9)
    if rank < 3:
        return False

    try:
        return float(mesh.convex_hull.volume) >= 1e-12
    except Exception:  # noqa: BLE001
        return False


def _export_visual_objs(
    src_mesh: Path,
    dst_dir: Path,
    base_name: str,
    scale: Tuple[float, float, float],
) -> List[Path]:
    """Export one OBJ/MTL for each visual geometry to preserve per-part colors."""
    if src_mesh.suffix.lower() == ".obj":
        dst_mesh = dst_dir / f"{base_name}.obj"
        dst_mesh.parent.mkdir(parents=True, exist_ok=True)
        return [_export_plain_obj(src_mesh, dst_mesh, scale, include_materials=True)]

    meshes = _load_meshes(src_mesh)
    if not meshes:
        raise RuntimeError(f"No valid mesh geometry in {src_mesh}")
    meshes = [_scaled_mesh(mesh, scale) for mesh in meshes]
    meshes = [mesh for mesh in meshes if _is_valid_visual_mesh(mesh)]
    if not meshes:
        raise RuntimeError(f"No valid visual mesh geometry in {src_mesh}")

    out_files: List[Path] = []
    for i, mesh in enumerate(meshes):
        output_name = f"{base_name}.obj" if len(meshes) == 1 else f"{base_name}_visual{i:03d}.obj"
        dst_mesh = dst_dir / output_name
        _write_obj_with_unique_assets(mesh, dst_mesh, include_materials=True)
        out_files.append(dst_mesh)
    return out_files


def _parse_mesh_scale(scale: Optional[str]) -> Tuple[float, float, float]:
    if not scale:
        return (1.0, 1.0, 1.0)
    values = [float(v) for v in scale.split()]
    if len(values) != 3:
        return (1.0, 1.0, 1.0)
    return (values[0], values[1], values[2])


def _is_valid_collision_mesh(mesh: trimesh.Trimesh) -> bool:
    """Reject split collision pieces that are too flat or too small for MuJoCo/QHull."""
    if len(mesh.faces) < 4 or len(mesh.vertices) < 4:
        return False

    vertices = np.asarray(mesh.vertices, dtype=float)
    vertices = np.unique(vertices, axis=0)
    if len(vertices) < 4:
        return False

    extents = np.ptp(vertices, axis=0)
    rank_scale = max(float(extents.max()), 1.0)
    rank = np.linalg.matrix_rank(vertices - vertices.mean(axis=0), tol=rank_scale * 1e-9)
    if rank < 3:
        return False

    try:
        return float(mesh.convex_hull.volume) >= 1e-12
    except Exception:  # noqa: BLE001
        return False


def _split_obj_by_object(src_mesh: Path, dst_dir: Path, base_name: str, scale: Tuple[float, float, float]) -> List[Path]:
    """Split collision meshes into connected 3D pieces and drop degenerate pieces."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    loaded_meshes = _load_meshes(src_mesh)
    if not loaded_meshes:
        raise RuntimeError(f"No valid mesh geometry in {src_mesh}")

    meshes: List[trimesh.Trimesh] = []
    for mesh in loaded_meshes:
        mesh = _scaled_mesh(mesh, scale)
        parts = [part for part in mesh.split(only_watertight=False) if _is_valid_collision_mesh(part)]
        if 1 < len(parts) <= 64:
            meshes.extend(parts)
        elif _is_valid_collision_mesh(mesh):
            meshes.append(mesh)

    out_files: List[Path] = []
    for i, mesh in enumerate(meshes):
        out = dst_dir / f"{base_name}_collision{i:03d}.obj"
        _write_obj_with_unique_assets(mesh, out, include_materials=False)
        out_files.append(out)
    return out_files


def _rewrite_visual_meshes(
    root: ET.Element,
    src_urdf_dir: Path,
    dst_visual_dir: Path,
) -> Dict[str, List[str]]:
    remap: Dict[str, List[str]] = {}
    seen_outputs: Dict[str, str] = {}

    for link in root.findall(".//link"):
        new_children = []
        for child in list(link):
            if child.tag != "visual":
                new_children.append(child)
                continue

            geom = child.find("geometry")
            mesh = geom.find("mesh") if geom is not None else None
            src_rel = mesh.attrib.get("filename") if mesh is not None else None
            if not src_rel:
                new_children.append(child)
                continue

            if src_rel not in remap:
                src_abs = (src_urdf_dir / src_rel).resolve()
                base_name = Path(src_rel).stem
                mesh_scale = _parse_mesh_scale(mesh.attrib.get("scale") if mesh is not None else None)
                out_files = _export_visual_objs(src_abs, dst_visual_dir, base_name, mesh_scale)
                remap[src_rel] = []
                for out in out_files:
                    _check_unique_output(seen_outputs, src_rel, out.name)
                    remap[src_rel].append(str(Path("meshes") / "visual" / out.name))

            out_refs = remap[src_rel]
            link_name = link.attrib.get("name", "link")
            visual_base_name = Path(src_rel).stem
            if len(out_refs) == 1:
                child.attrib["name"] = f"{link_name}_{visual_base_name}_visual"
                mesh.attrib["filename"] = out_refs[0]
                mesh.attrib.pop("scale", None)
                new_children.append(child)
                continue

            for i, out_ref in enumerate(out_refs):
                visual = copy.deepcopy(child)
                visual.attrib["name"] = f"{link_name}_{visual_base_name}_visual{i:03d}"
                visual_mesh = visual.find("geometry").find("mesh")
                visual_mesh.attrib["filename"] = out_ref
                visual_mesh.attrib.pop("scale", None)
                new_children.append(visual)

        link[:] = new_children
    return remap


#############
# URDF mesh rewriting
#############
def _rewrite_collision_meshes(
    root: ET.Element,
    src_urdf_dir: Path,
    dst_collision_dir: Path,
) -> Tuple[int, int, int]:
    """Replace each collision mesh reference with exported OBJ collision pieces."""
    split_count = 0
    total_collision_mesh = 0
    skipped_collision_mesh = 0

    for link in root.findall(".//link"):
        new_children = []
        for child in list(link):
            if child.tag != "collision":
                new_children.append(child)
                continue

            geom = child.find("geometry")
            mesh = geom.find("mesh") if geom is not None else None
            src_rel = mesh.attrib.get("filename") if mesh is not None else None
            if not src_rel:
                new_children.append(child)
                continue

            total_collision_mesh += 1
            src_abs = (src_urdf_dir / src_rel).resolve()
            base_name = Path(src_rel).stem
            mesh_scale = _parse_mesh_scale(mesh.attrib.get("scale") if mesh is not None else None)
            out_files = _split_obj_by_object(src_abs, dst_collision_dir, base_name=base_name, scale=mesh_scale)
            split_count += len(out_files)

            if not out_files:
                skipped_collision_mesh += 1
                continue

            if len(out_files) == 1:
                child_mesh = child.find("geometry").find("mesh")
                child_mesh.attrib["filename"] = str(Path("meshes") / "collision" / out_files[0].name)
                child_mesh.attrib.pop("scale", None)
                new_children.append(child)
                continue

            for i, out in enumerate(out_files):
                c = copy.deepcopy(child)
                c_name = c.attrib.get("name", "collision")
                link_name = link.attrib.get("name", "link")
                c.attrib["name"] = f"{link_name}_{c_name}_part{i:03d}"
                c_mesh = c.find("geometry").find("mesh")
                c_mesh.attrib["filename"] = str(Path("meshes") / "collision" / out.name)
                c_mesh.attrib.pop("scale", None)
                new_children.append(c)

        link[:] = new_children

    return total_collision_mesh, split_count, skipped_collision_mesh


#############
# MuJoCo compilation and MJCF material patching
#############
def _try_compile_to_mjcf(src_urdf: Path, output_xml: Path) -> Optional[str]:
    try:
        import mujoco

        assets = {}
        for p in sorted((src_urdf.parent / "meshes").rglob("*")):
            if p.is_file():
                assets[p.relative_to(src_urdf.parent).as_posix()] = p.read_bytes()

        compile_urdf = _make_visual_preserving_urdf(src_urdf)
        try:
            model = mujoco.MjModel.from_xml_path(str(compile_urdf), assets=assets)
        finally:
            compile_urdf.unlink(missing_ok=True)
        mujoco.mj_saveLastXML(str(output_xml), model)
        _restore_mjcf_mesh_paths(output_xml)
        _patch_visual_materials(output_xml)
        return None
    except Exception as e:  # noqa: BLE001
        return str(e)


def _try_compile_xml(xml_path: Path) -> Optional[str]:
    try:
        import mujoco

        mujoco.MjModel.from_xml_path(str(xml_path))
        return None
    except Exception as e:  # noqa: BLE001
        return str(e)


def _make_visual_preserving_urdf(src_urdf: Path) -> Path:
    tree = ET.parse(src_urdf)
    root = tree.getroot()
    mujoco = root.find("mujoco")
    if mujoco is None:
        mujoco = ET.Element("mujoco")
        root.insert(0, mujoco)
    compiler = mujoco.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco, "compiler")
    compiler.attrib["discardvisual"] = "false"

    with tempfile.NamedTemporaryFile(dir=src_urdf.parent, suffix=".urdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    ET.indent(tree, space="  ")
    tree.write(tmp_path, encoding="utf-8", xml_declaration=True)
    return tmp_path


def _restore_mjcf_mesh_paths(output_xml: Path) -> None:
    tree = ET.parse(output_xml)
    root = tree.getroot()
    for mesh in root.findall(".//asset/mesh"):
        filename = mesh.attrib.get("file")
        if not filename:
            continue
        if (output_xml.parent / filename).exists():
            continue
        if (output_xml.parent / "meshes" / "collision" / filename).exists():
            mesh.attrib["file"] = str(Path("meshes") / "collision" / filename)
        elif (output_xml.parent / "meshes" / "visual" / filename).exists():
            mesh.attrib["file"] = str(Path("meshes") / "visual" / filename)
    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="utf-8")


def _parse_mtl(mtl_path: Path, material_name: str) -> Tuple[Optional[str], Optional[str]]:
    if not mtl_path.exists():
        return None, None

    active = False
    kd: Optional[str] = None
    texture: Optional[str] = None
    for line in mtl_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "newmtl":
            active = len(parts) > 1 and parts[1] == material_name
            continue
        if not active:
            continue
        if parts[0] == "Kd" and len(parts) >= 4:
            kd = " ".join(parts[1:4])
        elif parts[0].startswith("map_") and len(parts) >= 2:
            texture = parts[-1]
    return kd, texture


def _obj_material(obj_path: Path) -> Tuple[Optional[Path], Optional[str]]:
    mtl_path: Optional[Path] = None
    material_name: Optional[str] = None
    for line in obj_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "mtllib" and len(parts) >= 2:
            mtl_path = obj_path.parent / parts[-1]
        elif parts[0] == "usemtl" and len(parts) >= 2:
            material_name = parts[1]
            break
    return mtl_path, material_name


def _patch_visual_materials(output_xml: Path) -> None:
    tree = ET.parse(output_xml)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    mesh_to_material: Dict[str, str] = {}
    for mesh in root.findall(".//asset/mesh"):
        mesh_file = mesh.attrib.get("file")
        mesh_name = mesh.attrib.get("name")
        if not mesh_file or not mesh_name or not mesh_file.startswith("meshes/visual/"):
            continue

        obj_path = output_xml.parent / mesh_file
        mtl_path, source_material = _obj_material(obj_path)
        if mtl_path is None or source_material is None:
            continue

        kd, texture = _parse_mtl(mtl_path, source_material)
        material_name = f"{mesh_name}_material"
        material_attrs = {"name": material_name}

        if texture is not None and (mtl_path.parent / texture).exists():
            texture_name = f"{mesh_name}_texture"
            ET.SubElement(
                asset,
                "texture",
                name=texture_name,
                type="2d",
                file=str(Path("meshes") / "visual" / texture),
            )
            material_attrs["texture"] = texture_name
        if kd is not None:
            material_attrs["rgba"] = f"{kd} 1"

        ET.SubElement(asset, "material", **material_attrs)
        mesh_to_material[mesh_name] = material_name

    if not mesh_to_material:
        return

    for geom in root.findall(".//geom"):
        mesh_name = geom.attrib.get("mesh")
        is_visual = geom.attrib.get("contype") == "0" and geom.attrib.get("conaffinity") == "0"
        if is_visual and mesh_name in mesh_to_material:
            geom.attrib["material"] = mesh_to_material[mesh_name]

    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="utf-8")


def _write_collision_only_mjcf(src_xml: Path, dst_xml: Path) -> None:
    """Write a companion MJCF that hides visual geoms and colors collision geoms."""
    tree = ET.parse(src_xml)
    root = tree.getroot()

    for parent in root.iter():
        for child in list(parent):
            if child.tag != "geom":
                continue
            is_visual = child.attrib.get("contype") == "0" and child.attrib.get("conaffinity") == "0"
            if is_visual:
                parent.remove(child)

    referenced_meshes = set()
    for geom in root.findall(".//geom"):
        geom.attrib["group"] = "1"
        geom.attrib.pop("material", None)
        geom.attrib.setdefault("rgba", "0.9 0.35 0.1 0.65")
        mesh_name = geom.attrib.get("mesh")
        if mesh_name:
            referenced_meshes.add(mesh_name)

    asset = root.find("asset")
    if asset is not None:
        for child in list(asset):
            if child.tag == "material":
                asset.remove(child)
            elif child.tag == "mesh" and child.attrib.get("name") not in referenced_meshes:
                asset.remove(child)

    ET.indent(tree, space="  ")
    tree.write(dst_xml, encoding="utf-8")


#############
# URDF normalization
#############
def _link_has_geometry(link: ET.Element) -> bool:
    return link.find("visual") is not None or link.find("collision") is not None


def _rename_link_references(root: ET.Element, old_name: str, new_name: str) -> None:
    for parent in root.findall(".//joint/parent"):
        if parent.attrib.get("link") == old_name:
            parent.attrib["link"] = new_name
    for child in root.findall(".//joint/child"):
        if child.attrib.get("link") == old_name:
            child.attrib["link"] = new_name


def _root_links(root: ET.Element) -> List[str]:
    links = {link.attrib["name"] for link in root.findall("./link") if link.attrib.get("name")}
    children = {child.attrib["link"] for child in root.findall("./joint/child") if child.attrib.get("link")}
    return sorted(links - children)


def _normalize_hand_root(root: ET.Element) -> str:
    """Ensure the prepared URDF has a fixed hand_root -> model_root joint."""
    links = {link.attrib.get("name"): link for link in root.findall("./link") if link.attrib.get("name")}
    root_link_names = set(_root_links(root))

    for joint in root.findall("./joint"):
        if joint.attrib.get("type") != "fixed":
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        parent_name = parent.attrib.get("link") if parent is not None else None
        child_name = child.attrib.get("link") if child is not None else None
        parent_link = links.get(parent_name)
        child_link = links.get(child_name)
        if parent_name not in root_link_names or parent_link is None or child_link is None:
            continue
        if _link_has_geometry(parent_link) or parent_link.find("inertial") is not None:
            continue
        if not _link_has_geometry(child_link):
            continue

        if parent_name != "hand_root":
            parent_link.attrib["name"] = "hand_root"
            _rename_link_references(root, parent_name, "hand_root")
        joint.attrib["name"] = "hand_root_joint"
        return "renamed"

    root_links = _root_links(root)
    if not root_links:
        return "none"

    hand_root = ET.Element("link", name="hand_root")
    hand_root_joint = ET.Element("joint", name="hand_root_joint", type="fixed")
    ET.SubElement(hand_root_joint, "origin", rpy="0 0 0", xyz="0 0 0")
    ET.SubElement(hand_root_joint, "parent", link="hand_root")
    ET.SubElement(hand_root_joint, "child", link=root_links[0])

    root.insert(0, hand_root)
    root.insert(1, hand_root_joint)
    return "added"


def _parse_xyz(value: Optional[str]) -> np.ndarray:
    if not value:
        return np.zeros(3)
    parts = [float(v) for v in value.split()]
    if len(parts) != 3:
        return np.zeros(3)
    return np.asarray(parts, dtype=float)


def _collision_bounds(collision: ET.Element, asset_root: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    origin = _parse_xyz(collision.find("origin").attrib.get("xyz") if collision.find("origin") is not None else None)
    geom = collision.find("geometry")
    if geom is None:
        return None

    extents: Optional[np.ndarray] = None
    box = geom.find("box")
    sphere = geom.find("sphere")
    cylinder = geom.find("cylinder")
    mesh = geom.find("mesh")

    if box is not None and box.attrib.get("size"):
        extents = _parse_xyz(box.attrib["size"])
    elif sphere is not None and sphere.attrib.get("radius"):
        radius = float(sphere.attrib["radius"])
        extents = np.asarray([2 * radius, 2 * radius, 2 * radius], dtype=float)
    elif cylinder is not None and cylinder.attrib.get("radius") and cylinder.attrib.get("length"):
        radius = float(cylinder.attrib["radius"])
        length = float(cylinder.attrib["length"])
        extents = np.asarray([2 * radius, 2 * radius, length], dtype=float)
    elif mesh is not None and mesh.attrib.get("filename"):
        mesh_path = asset_root / mesh.attrib["filename"]
        if mesh_path.exists():
            loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
            if isinstance(loaded, trimesh.Trimesh) and len(loaded.vertices) > 0:
                scale = np.asarray(_parse_mesh_scale(mesh.attrib.get("scale")), dtype=float)
                extents = np.asarray(loaded.extents, dtype=float) * scale

    if extents is None or np.any(extents <= 0):
        return None

    half = extents / 2.0
    return origin - half, origin + half


def _ensure_missing_inertials(root: ET.Element, asset_root: Path) -> int:
    """Add conservative inertials only for physical links that have collision geometry."""
    added = 0
    for link in root.findall("./link"):
        if link.attrib.get("name") == "world" or link.find("inertial") is not None:
            continue

        bounds = [
            bound
            for collision in link.findall("collision")
            if (bound := _collision_bounds(collision, asset_root)) is not None
        ]
        if not bounds:
            continue

        mins = np.vstack([bound[0] for bound in bounds]).min(axis=0)
        maxs = np.vstack([bound[1] for bound in bounds]).max(axis=0)
        center = (mins + maxs) / 2.0
        extents = np.maximum(maxs - mins, 1e-3)

        volume = float(np.prod(extents))
        mass = min(max(volume * 1000.0, 1e-4), 0.05)
        inertia = np.maximum(
            mass
            / 12.0
            * np.asarray(
                [
                    extents[1] ** 2 + extents[2] ** 2,
                    extents[0] ** 2 + extents[2] ** 2,
                    extents[0] ** 2 + extents[1] ** 2,
                ]
            ),
            1e-8,
        )

        inertial = ET.Element("inertial")
        ET.SubElement(inertial, "origin", rpy="0 0 0", xyz=f"{center[0]:.9g} {center[1]:.9g} {center[2]:.9g}")
        ET.SubElement(inertial, "mass", value=f"{mass:.9g}")
        ET.SubElement(
            inertial,
            "inertia",
            ixx=f"{inertia[0]:.9g}",
            ixy="0",
            ixz="0",
            iyy=f"{inertia[1]:.9g}",
            iyz="0",
            izz=f"{inertia[2]:.9g}",
        )
        link.insert(0, inertial)
        added += 1
    return added


def _ensure_mujoco_compiler(root: ET.Element) -> None:
    for existing in list(root.findall("mujoco")):
        root.remove(existing)

    mujoco = ET.Element("mujoco")
    ET.SubElement(
        mujoco,
        "compiler",
        balanceinertia="true",
        discardvisual="false",
        fusestatic="false",
        inertiafromgeom="false",
    )
    root.insert(0, mujoco)


#############
# Output cleanup and migration orchestration
#############
def _output_stem(src_urdf: Path) -> str:
    stem = src_urdf.stem
    if stem.endswith("_glb"):
        return stem[: -len("_glb")]
    return stem


def _asset_refs_from_xml(path: Path) -> List[Path]:
    if not path.exists():
        return []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return []

    refs: List[Path] = []
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename") or mesh.attrib.get("file")
        if filename:
            refs.append(Path(filename))
    for texture in root.findall(".//texture"):
        filename = texture.attrib.get("file")
        if filename:
            refs.append(Path(filename))
    return refs


def _write_elementtree_copy(src_xml: Path, dst_xml: Path) -> None:
    """Save the source URDF after ElementTree parsing/formatting for diffing."""
    tree = ET.parse(src_xml)
    ET.indent(tree, space="  ")
    tree.write(dst_xml, encoding="utf-8", xml_declaration=True)


def _clean_generated_outputs(dst_root: Path, output_stem: str, source_stem: str, side: str) -> None:
    """Remove files referenced by a previous run for the same source before regenerating."""
    stems = {output_stem, source_stem}
    metadata_paths = [
        p
        for stem in stems
        for p in (
            dst_root / f"{stem}.xml",
            dst_root / f"{stem}_collision.xml",
            dst_root / f"{stem}_for_mjcf.urdf",
            dst_root / f"{stem}_original.urdf",
        )
    ]

    for ref in {ref for path in metadata_paths for ref in _asset_refs_from_xml(path)}:
        asset_path = dst_root / ref
        if asset_path.is_file():
            asset_path.unlink()

    for directory in [dst_root / "meshes" / "visual", dst_root / "meshes" / "collision"]:
        if not directory.exists():
            continue
        for path in directory.glob(f"{side}_*"):
            if path.is_file():
                path.unlink()

    for path in metadata_paths:
        if path.exists():
            path.unlink()


def _process_side(cfg: Config, repo_root: Path, hand_dir: Path, side: str) -> None:
    """Run the full URDF asset rewrite, MuJoCo compile, and MJCF post-process pipeline."""
    src_urdf = _pick_source_urdf(hand_dir, cfg.hand_name, side)

    dst_root = (repo_root / cfg.output_root / cfg.hand_name).resolve()
    dst_visual_dir = dst_root / "meshes" / "visual"
    dst_collision_dir = dst_root / "meshes" / "collision"
    dst_root.mkdir(parents=True, exist_ok=True)
    output_stem = _output_stem(src_urdf)
    if cfg.clean_output:
        _clean_generated_outputs(dst_root, output_stem, src_urdf.stem, side)

    original_urdf = dst_root / f"{output_stem}_original.urdf"
    _write_elementtree_copy(src_urdf, original_urdf)

    tree = ET.parse(src_urdf)
    root = tree.getroot()
    _ensure_mujoco_compiler(root)
    root_status = _normalize_hand_root(root)

    visual_remap = _rewrite_visual_meshes(root, src_urdf.parent, dst_visual_dir)
    total_collision_mesh, split_count, skipped_collision_mesh = _rewrite_collision_meshes(
        root,
        src_urdf.parent,
        dst_collision_dir,
    )
    added_inertials = _ensure_missing_inertials(root, dst_root)

    prepared_urdf = dst_root / f"{output_stem}_for_mjcf.urdf"
    ET.indent(tree, space="  ")
    tree.write(prepared_urdf, encoding="utf-8", xml_declaration=True)

    print(f"[done] source urdf: {src_urdf}")
    print(f"[done] side       : {side}")
    print(f"[done] output dir : {dst_root}")
    print(f"[done] original urdf copy: {original_urdf}")
    print(f"[done] hand root normalization: {root_status}")
    print(f"[done] visual remap count: {len(visual_remap)}")
    print(f"[done] collision mesh tags: {total_collision_mesh}, exported collision parts: {split_count}")
    if skipped_collision_mesh:
        print(f"[done] skipped degenerate collision meshes: {skipped_collision_mesh}")
    if added_inertials:
        print(f"[done] added missing inertials: {added_inertials}")
    print(f"[done] prepared urdf: {prepared_urdf}")

    if cfg.try_compile_mjcf:
        out_xml = dst_root / f"{output_stem}.xml"
        err = _try_compile_to_mjcf(prepared_urdf, out_xml)
        if err is None:
            print(f"[done] mjcf compiled: {out_xml}")
            can_write_collision_xml = True
            if cfg.patch_mjcf:
                patch_mjcf(prepared_urdf, out_xml, cfg.kp)
                patched_err = _try_compile_xml(out_xml)
                if patched_err is None:
                    print(f"[done] mjcf patched and verified: {out_xml}")
                else:
                    can_write_collision_xml = False
                    print(f"[warn] patched mjcf compile failed: {patched_err.splitlines()[0]}")
            if can_write_collision_xml:
                collision_xml = dst_root / f"{output_stem}_collision.xml"
                _write_collision_only_mjcf(out_xml, collision_xml)
                collision_err = _try_compile_xml(collision_xml)
                if collision_err is None:
                    print(f"[done] collision-only mjcf: {collision_xml}")
                else:
                    print(f"[warn] collision-only mjcf compile failed: {collision_err.splitlines()[0]}")
        else:
            print(f"[warn] mjcf compile failed: {err.splitlines()[0]}")


def main(cfg: Config) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    hand_dir = repo_root / "robots" / "hands" / cfg.hand_name
    if not hand_dir.exists():
        raise FileNotFoundError(f"Hand directory not found: {hand_dir}")

    sides = _sides_to_process(cfg, hand_dir)
    print(f"[info] sides to process: {', '.join(sides)}")
    for side in sides:
        _process_side(cfg, repo_root, hand_dir, side)


#############
# CLI
#############
def _parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Migrate a hand URDF into MJCF-friendly assets")
    parser.add_argument("--hand-name", required=True)
    parser.add_argument("--side", default="all", choices=["all", "right", "left"])
    parser.add_argument("--output-root", default="robots_mjcf/hands")
    parser.add_argument(
        "--clean-output",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--try-compile-mjcf",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--patch-mjcf",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--kp", type=float, default=1.0)
    args = parser.parse_args()
    return Config(
        hand_name=args.hand_name,
        side=args.side,
        output_root=args.output_root,
        clean_output=args.clean_output,
        try_compile_mjcf=args.try_compile_mjcf,
        patch_mjcf=args.patch_mjcf,
        kp=args.kp,
    )


if __name__ == "__main__":
    main(_parse_args())
