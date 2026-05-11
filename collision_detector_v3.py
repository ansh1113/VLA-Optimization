import coal
"""
Production Collision Detector V3 - FIXED URDF Parsing & Auto-Filtering

Properly maps URDF links to Pinocchio joint frames and automatically
filters out false-positive collisions using Kinematic Distance.

URDF structure: <link> elements contain geometry
Pinocchio structure: Joints connect links, frames track link placements
"""

import numpy as np
import pinocchio as pin
import coal
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import xml.etree.ElementTree as ET
import warnings

# Suppress Pinocchio deprecation warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning, module="pinocchio")

class CollisionDetectorV3:
    """
    Production collision detection with proper URDF→Pinocchio mapping.
    """
    
    def __init__(
        self,
        urdf_path: Union[str, Path],
        safety_margin: float = 0.0,
        verbose: bool = False
    ):
        urdf_path = Path(urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        
        self.urdf_path = urdf_path
        self.urdf_dir = urdf_path.parent
        self.safety_margin = safety_margin
        self.verbose = verbose
        
        # Build Pinocchio model
        try:
            self.model = pin.buildModelFromUrdf(str(urdf_path))
            self.data = self.model.createData()
        except Exception as e:
            raise RuntimeError(f"Failed to parse URDF: {e}")
        
        self.nq = self.model.nq
        
        # Build link→joint mapping from URDF
        self.link_to_joint = self._build_link_to_joint_mapping()
        
        # Parse URDF and build collision model
        self.collision_model, self.collision_data = self._parse_urdf_and_build_collision_model()
        
        self.component_to_links: Dict[str, List[int]] = {}
        
        if self.verbose:
            print(f"CollisionDetectorV3:")
            print(f"  Robot: {urdf_path.name}")
            print(f"  DOF: {self.nq}")
            print(f"  Collision objects: {len(self.collision_model.geometryObjects)}")
            print(f"  Active collision pairs: {len(self.collision_model.collisionPairs)}")
    
    def _build_link_to_joint_mapping(self) -> Dict[str, int]:
        """
        Build mapping from URDF link names to Pinocchio joint IDs.
        """
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        
        link_to_joint = {}
        
        for joint_elem in root.findall('joint'):
            joint_name = joint_elem.get('name')
            child_elem = joint_elem.find('child')
            if child_elem is not None:
                child_link = child_elem.get('link')
                for joint_id in range(self.model.njoints):
                    if self.model.names[joint_id] == joint_name:
                        link_to_joint[child_link] = joint_id
                        break
        
        robot_elem = root.find('robot')
        if robot_elem is not None:
            all_child_links = set()
            for joint_elem in root.findall('joint'):
                child_elem = joint_elem.find('child')
                if child_elem is not None:
                    all_child_links.add(child_elem.get('link'))
            
            for link_elem in root.findall('link'):
                link_name = link_elem.get('name')
                if link_name not in all_child_links:
                    link_to_joint[link_name] = 0  # universe frame
        
        return link_to_joint
    
    def _resolve_mesh_path(self, mesh_filename: str) -> Optional[Path]:
        """Resolve mesh file path."""
        if mesh_filename.startswith('package://'):
            parts = mesh_filename.replace('package://', '').split('/')
            for i in range(len(parts)):
                relative_path = '/'.join(parts[i:])
                for search_dir in [self.urdf_dir, self.urdf_dir.parent, self.urdf_dir.parent.parent]:
                    candidate = search_dir / relative_path
                    if candidate.exists():
                        return candidate
                    candidate = search_dir / 'meshes' / relative_path
                    if candidate.exists():
                        return candidate
        
        candidate = Path(mesh_filename)
        if candidate.exists(): return candidate
        candidate = self.urdf_dir / mesh_filename
        if candidate.exists(): return candidate
        
        for base in [self.urdf_dir, self.urdf_dir.parent]:
            for subdir in ['meshes', 'mesh', 'visual', 'collision']:
                candidate = base / subdir / Path(mesh_filename).name
                if candidate.exists(): return candidate
        
        return None

    def _get_kinematic_distance(self, j1: int, j2: int) -> int:
        """Finds shortest path distance between two joints in the kinematic tree."""
        if j1 == j2:
            return 0
            
        path1 = {j1: 0}
        curr = j1
        dist = 0
        while curr > 0:
            curr = self.model.parents[curr]
            dist += 1
            path1[curr] = dist
            
        curr = j2
        dist = 0
        while curr > 0:
            if curr in path1:
                return dist + path1[curr]
            curr = self.model.parents[curr]
            dist += 1
            
        return float('inf')
    
    def _parse_urdf_and_build_collision_model(self) -> Tuple:
        """Parse URDF and build HPP-FCL collision model."""
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        
        collision_model = pin.GeometryModel()
        geom_count = 0
        
        # Parse all links
        for link_elem in root.findall('link'):
            link_name = link_elem.get('name')
            if link_name not in self.link_to_joint:
                if self.verbose: print(f"  Warning: Link '{link_name}' not mapped to joint, skipping")
                continue
            
            joint_id = self.link_to_joint[link_name]
            
            collision_elem = link_elem.find('collision')
            if collision_elem is None:
                collision_elem = link_elem.find('visual')
            if collision_elem is None:
                continue
            
            geometry_elem = collision_elem.find('geometry')
            if geometry_elem is None:
                continue
            
            origin_elem = collision_elem.find('origin')
            if origin_elem is not None:
                xyz = np.array([float(x) for x in origin_elem.get('xyz', '0 0 0').split()])
                rpy = np.array([float(x) for x in origin_elem.get('rpy', '0 0 0').split()])
                placement = pin.SE3.Identity()
                placement.translation = xyz
                placement.rotation = pin.rpy.rpyToMatrix(rpy[0], rpy[1], rpy[2])
            else:
                placement = pin.SE3.Identity()
            
            coal_geom = None
            
            box = geometry_elem.find('box')
            if box is not None:
                size = np.array([float(x) for x in box.get('size').split()])
                coal_geom = coal.Box(size[0], size[1], size[2])
            
            cylinder = geometry_elem.find('cylinder')
            if cylinder is not None:
                coal_geom = coal.Cylinder(float(cylinder.get('radius')), float(cylinder.get('length')))
            
            sphere = geometry_elem.find('sphere')
            if sphere is not None:
                coal_geom = coal.Sphere(float(sphere.get('radius')))
            
            mesh = geometry_elem.find('mesh')
            if mesh is not None:
                filename = mesh.get('filename')
                resolved_path = self._resolve_mesh_path(filename)
                if resolved_path is not None:
                    try:
                        coal_geom = coal.MeshLoader().load(str(resolved_path))
                    except Exception as e:
                        if self.verbose: print(f"  Mesh load failed: {link_name} - {e}")
                        coal_geom = None
            
            if coal_geom is None:
                inertia = self.model.inertias[joint_id]
                if inertia.mass > 1e-6:
                    avg_I = np.mean(inertia.inertia.diagonal())
                    radius = np.sqrt(5 * avg_I / (2 * inertia.mass)) if avg_I > 0 else 0.05
                    coal_geom = coal.Sphere(np.clip(radius, 0.02, 0.5))
            
            if coal_geom is not None:
                # Pinocchio 3 style: Swap placement and geometry
                geom_obj = pin.GeometryObject(f"{link_name}_collision", joint_id, placement, coal_geom)    
                collision_model.addGeometryObject(geom_obj)
                geom_count += 1
        
        # --- NEW: EXPLICIT WHITELIST PAIRING ---
        # Instead of adding all and removing (which bugs out in Pinocchio Python),
        # we start empty and explicitly ONLY pair geometries that are >2 joints apart.
        n_geoms = len(collision_model.geometryObjects)
        for i in range(n_geoms):
            for j in range(i + 1, n_geoms):
                geom1 = collision_model.geometryObjects[i]
                geom2 = collision_model.geometryObjects[j]
                
                parent1 = geom1.parentJoint
                parent2 = geom2.parentJoint
                
                if self._get_kinematic_distance(parent1, parent2) > 2:
                    collision_model.addCollisionPair(pin.CollisionPair(i, j))
        
        collision_data = collision_model.createData()
        return collision_model, collision_data
    
    def set_component_mapping(self, component_map: Dict[str, List[int]]):
        self.component_to_links = component_map.copy()
    
    def check_self_collision(self, q: np.ndarray) -> Tuple[bool, List[Dict]]:
        if q.shape[0] != self.nq:
            raise ValueError(f"Expected q of size {self.nq}, got {q.shape[0]}")
        
        pin.updateGeometryPlacements(self.model, self.data, self.collision_model, self.collision_data, q)
        pin.computeCollisions(self.model, self.data, self.collision_model, self.collision_data, q, False)
        
        collisions = []
        for pair_id in range(len(self.collision_model.collisionPairs)):
            pair = self.collision_model.collisionPairs[pair_id]
            result = self.collision_data.collisionResults[pair_id]
            
            if result.isCollision():
                geom1 = self.collision_model.geometryObjects[pair.first]
                geom2 = self.collision_model.geometryObjects[pair.second]
                
                M1 = self.collision_data.oMg[pair.first]
                M2 = self.collision_data.oMg[pair.second]
                T1 = coal.Transform3s(M1.rotation, M1.translation)
                T2 = coal.Transform3s(M2.rotation, M2.translation)
                
                distance = coal.distance(
                    geom1.geometry, T1,
                    geom2.geometry, T2,
                    coal.DistanceRequest(), coal.DistanceResult()
                )
                
                if distance < self.safety_margin:
                    collisions.append({
                        'geom1': geom1.name,
                        'geom2': geom2.name,
                        'distance': distance,
                        'penetration': -distance if distance < 0 else 0.0
                    })
        
        return len(collisions) > 0, collisions
    
    def get_collision_report(self, q: np.ndarray) -> Dict:
        has_collision, collisions = self.check_self_collision(q)
        return {
            'has_collision': has_collision,
            'num_collisions': len(collisions),
            'collisions': collisions,
            'is_safe': not has_collision,
            'uses_coal': True
        }


if __name__ == "__main__":
    print("="*80)
    print("COLLISION DETECTOR V3 - FIXED MAPPING")
    print("="*80)
    
    urdf = "/scratch/anshb3/ovla/robots/unitree_ros/robots/g1_description/g1_23dof.urdf"
    
    try:
        detector = CollisionDetectorV3(urdf, safety_margin=0.05, verbose=True)
        q_neutral = np.zeros(detector.nq)
        report = detector.get_collision_report(q_neutral)
        print(f"\nHas collision: {report['has_collision']}")
        print("\n✅ COLLISION DETECTOR V3 WORKING")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
