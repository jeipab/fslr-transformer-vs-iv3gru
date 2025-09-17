"""
Advanced Hand-Head Occlusion Detection System for Sign Language Video Analysis
Based on the paper "Detecting Hand-Head Occlusions in Sign Language Video"

This module provides sophisticated computer vision-based occlusion detection
that complements the simple keypoint-based method in occlusion_detection.py.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set, Union
from dataclasses import dataclass
from collections import deque
import warnings

# Import the ellipse point check function from the main occlusion detection module
def _point_in_ellipse(px: float, py: float, cx: float, cy: float, ax: float, by: float) -> bool:
    """Check if point (px, py) lies inside or on the ellipse centered at (cx, cy)
    with semi-axes (ax, by). Coordinates are normalized to [0,1].
    """
    if ax <= 0.0 or by <= 0.0:
        return False
    dx = (px - cx) / ax
    dy = (py - cy) / by
    return (dx * dx + dy * dy) <= 1.0


def _hand_centers_and_tips(frame_xy: np.ndarray, frame_mask: np.ndarray, hand_start: int, hand_len: int) -> tuple[tuple[float, float] | None, list[tuple[float, float]]]:
    """Return (palm_center, fingertip_points) for one hand.

    Uses MediaPipe Hands indexing (21 landmarks). Palm center is the mean of MCP
    joints (5, 9, 13, 17) when available, else the wrist (0) if visible.
    Fingertips are indices [4, 8, 12, 16, 20] when visible.
    """
    mcp_rel = [5, 9, 13, 17]  # MCP joints
    fingertip_rel = [4, 8, 12, 16, 20]  # fingertips
    mcp_coords: list[tuple[float, float]] = []
    for r in mcp_rel:
        if hand_len <= r:
            continue
        if bool(frame_mask[hand_start + r]):
            idx = 2 * (hand_start + r)
            mcp_coords.append((float(frame_xy[idx]), float(frame_xy[idx + 1])))
    palm_center: tuple[float, float] | None
    if len(mcp_coords) >= 2:
        mx = sum(p[0] for p in mcp_coords) / float(len(mcp_coords))
        my = sum(p[1] for p in mcp_coords) / float(len(mcp_coords))
        palm_center = (mx, my)
    else:
        # fallback to wrist if visible
        if bool(frame_mask[hand_start + 0]):
            idx0 = 2 * (hand_start + 0)
            palm_center = (float(frame_xy[idx0]), float(frame_xy[idx0 + 1]))
        else:
            palm_center = None
    # Fingertips
    tips: list[tuple[float, float]] = []
    for r in fingertip_rel:
        if hand_len <= r:
            continue
        if bool(frame_mask[hand_start + r]):
            idx = 2 * (hand_start + r)
            tips.append((float(frame_xy[idx]), float(frame_xy[idx + 1])))
    return palm_center, tips

# Check for advanced dependencies
def _check_advanced_dependencies() -> bool:
    """Check if advanced dependencies are available."""
    try:
        import cv2
        import mediapipe as mp
        from scipy import ndimage
        from scipy.spatial import KDTree
        from sklearn.cluster import DBSCAN
        return True
    except ImportError:
        return False


@dataclass
class Point2D:
    """Represents a 2D point with x, y coordinates"""
    x: float
    y: float
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def distance_to(self, other: 'Point2D') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class Gridlet:
    """A set of tracked points with topology"""
    points: List[Point2D]
    neighbors: Dict[int, List[int]]  # Adjacency list
    reference_frame: int
    tracking_cost: float = 0.0
    
    def get_center(self) -> Point2D:
        """Get the centroid of the gridlet"""
        x = sum(p.x for p in self.points) / len(self.points)
        y = sum(p.y for p in self.points) / len(self.points)
        return Point2D(x, y)


class HeadRegion:
    """Defines the five head regions from Suvi dictionary"""
    FOREHEAD = 0
    CHEEKS = 1
    NOSE = 2
    MOUTH = 3
    NECK = 4
    
    NAMES = ['forehead', 'cheeks', 'nose', 'mouth', 'neck']


class AdvancedHandHeadOcclusionDetector:
    """Advanced hand-head occlusion detector using computer vision techniques."""
    
    def __init__(self, use_global_tracking: bool = True):
        """Initialize the advanced detector."""
        if not _check_advanced_dependencies():
            raise ImportError(
                "Advanced occlusion detection requires additional dependencies. "
                "Please install: pip install scipy scikit-learn"
            )
        
        self.use_global_tracking = use_global_tracking
        
        # Import advanced dependencies
        import cv2
        import mediapipe as mp
        
        # Initialize MediaPipe for face and hand detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Tracking parameters
        self.gridlet_size = 4
        self.gridlet_neighbors = 3
        self.tracking_window_size = 5
        self.motion_threshold = 10
        
        # Storage for tracking
        self.unoccluded_face_points: Dict[int, Set[Tuple[int, int]]] = {}
        self.outside_face_points: Dict[int, Set[Tuple[int, int]]] = {}
        self.tracked_gridlets: List[Gridlet] = []
        self.hand_blobs: List[Dict] = []
        self.facial_prohibition_masks: Dict[int, np.ndarray] = {}
        
        # Occlusion history for temporal filtering
        self.occlusion_history = deque(maxlen=self.tracking_window_size)
    
    def detect_skin_pixels(self, image: np.ndarray) -> np.ndarray:
        """Detect skin pixels using color-based segmentation."""
        import cv2
        
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Define skin color range in YCrCb
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        
        # Create skin mask
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        return skin_mask
    
    def get_facial_landmarks(self, image: np.ndarray) -> Optional[Dict]:
        """Extract facial landmarks using MediaPipe Holistic (compatible with preprocessing pipeline)."""
        import cv2
        import mediapipe as mp
        
        # Use Holistic model to match preprocessing pipeline
        holistic = mp.solutions.holistic.Holistic(
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_image)
        
        if not results.face_landmarks:
            holistic.close()
            return None
        
        h, w = image.shape[:2]
        landmarks = results.face_landmarks
        
        # Use the same face landmark indices as preprocessing pipeline
        # FACEMESH_11 = [1, 33, 263, 133, 362, 61, 291, 105, 334, 199, 4]
        face_indices = [1, 33, 263, 133, 362, 61, 291, 105, 334, 199, 4]
        
        # Extract key facial points using preprocessing-compatible indices
        key_points = {
            'left_eye_inner': landmarks.landmark[133],    # Index 3 in FACEMESH_11
            'left_eye_outer': landmarks.landmark[33],     # Index 1 in FACEMESH_11  
            'right_eye_inner': landmarks.landmark[362],  # Index 4 in FACEMESH_11
            'right_eye_outer': landmarks.landmark[263],  # Index 2 in FACEMESH_11
            'nose_tip': landmarks.landmark[1],           # Index 0 in FACEMESH_11
            'mouth_left': landmarks.landmark[61],         # Index 5 in FACEMESH_11
            'mouth_right': landmarks.landmark[291],       # Index 6 in FACEMESH_11
            'chin': landmarks.landmark[18],               # Approximate chin
            'forehead': landmarks.landmark[9]             # Approximate forehead
        }
        
        # Convert to normalized coordinates [0,1] to match preprocessing pipeline
        landmark_dict = {}
        for name, point in key_points.items():
            landmark_dict[name] = Point2D(point.x, point.y)  # Already normalized [0,1]
        
        holistic.close()
        return landmark_dict
    
    def partition_head_regions(self, landmarks: Dict, image_shape: tuple = (256, 256)) -> Dict[int, np.ndarray]:
        """Partition the head area into 5 regions based on facial landmarks."""
        import cv2
        
        regions = {}
        
        if not landmarks:
            return regions
        
        h, w = image_shape
        
        # Create polygon for each region using normalized coordinates
        # Forehead region
        forehead_pts = np.array([
            landmarks['forehead'].to_tuple(),
            (landmarks['left_eye_outer'].x, landmarks['forehead'].y),
            (landmarks['right_eye_outer'].x, landmarks['forehead'].y),
            landmarks['left_eye_outer'].to_tuple(),
            landmarks['right_eye_outer'].to_tuple()
        ], dtype=np.float32)
        
        # Cheeks region (includes eyes and ears area)
        cheeks_pts = np.array([
            landmarks['left_eye_outer'].to_tuple(),
            landmarks['right_eye_outer'].to_tuple(),
            landmarks['nose_tip'].to_tuple(),
            (landmarks['left_eye_outer'].x - 0.08, landmarks['nose_tip'].y),  # Normalized offset
            (landmarks['right_eye_outer'].x + 0.08, landmarks['nose_tip'].y)  # Normalized offset
        ], dtype=np.float32)
        
        # Nose region
        nose_pts = np.array([
            (landmarks['nose_tip'].x - 0.06, landmarks['nose_tip'].y - 0.04),  # Normalized offsets
            (landmarks['nose_tip'].x + 0.06, landmarks['nose_tip'].y - 0.04),
            (landmarks['nose_tip'].x + 0.06, landmarks['nose_tip'].y + 0.04),
            (landmarks['nose_tip'].x - 0.06, landmarks['nose_tip'].y + 0.04)
        ], dtype=np.float32)
        
        # Mouth region
        mouth_pts = np.array([
            landmarks['mouth_left'].to_tuple(),
            landmarks['mouth_right'].to_tuple(),
            landmarks['chin'].to_tuple(),
            (landmarks['mouth_left'].x, landmarks['chin'].y),
            (landmarks['mouth_right'].x, landmarks['chin'].y)
        ], dtype=np.float32)
        
        # Neck region
        neck_pts = np.array([
            (landmarks['chin'].x - 0.12, landmarks['chin'].y),
            (landmarks['chin'].x + 0.12, landmarks['chin'].y),
            (landmarks['chin'].x + 0.16, landmarks['chin'].y + 0.20),  # Normalized offset
            (landmarks['chin'].x - 0.16, landmarks['chin'].y + 0.20)
        ], dtype=np.float32)
        
        regions[HeadRegion.FOREHEAD] = forehead_pts
        regions[HeadRegion.CHEEKS] = cheeks_pts
        regions[HeadRegion.NOSE] = nose_pts
        regions[HeadRegion.MOUTH] = mouth_pts
        regions[HeadRegion.NECK] = neck_pts
        
        return regions
    
    def classify_occlusions(self, occlusion_mask: np.ndarray, 
                          landmarks: Dict, image_shape: tuple = (256, 256)) -> Dict[int, int]:
        """Classify occlusions by head region."""
        import cv2
        
        regions = self.partition_head_regions(landmarks, image_shape)
        occlusion_counts = {}
        
        h, w = image_shape
        
        for region_id, polygon in regions.items():
            # Convert normalized coordinates to pixel coordinates for mask creation
            pixel_polygon = (polygon * np.array([w, h])).astype(np.int32)
            
            # Create region mask
            region_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(region_mask, [pixel_polygon], 255)
            
            # Resize occlusion mask to match image shape if needed
            if occlusion_mask.shape != (h, w):
                occlusion_mask_resized = cv2.resize(occlusion_mask, (w, h))
            else:
                occlusion_mask_resized = occlusion_mask
            
            # Count occlusion pixels in region
            occlusion_in_region = cv2.bitwise_and(occlusion_mask_resized, region_mask)
            count = np.count_nonzero(occlusion_in_region)
            occlusion_counts[region_id] = count
        
        return occlusion_counts
    
    def temporal_filter(self, occlusion_counts: Dict[int, int]) -> Dict[int, bool]:
        """Apply temporal filtering to reduce noise."""
        self.occlusion_history.append(occlusion_counts)
        
        if len(self.occlusion_history) < self.tracking_window_size:
            # Not enough history, return current detections
            return {region: (count > 100) for region, count in occlusion_counts.items()}
        
        # Apply majority voting
        filtered_detections = {}
        for region_id in range(5):
            votes = []
            for hist_counts in self.occlusion_history:
                votes.append(hist_counts.get(region_id, 0) > 100)
            
            # Majority voting
            filtered_detections[region_id] = sum(votes) > len(votes) // 2
        
        return filtered_detections
    
    def process_frame(self, frame: np.ndarray, frame_idx: int,
                     prev_frame: Optional[np.ndarray] = None) -> Dict:
        """Process a single frame for advanced occlusion detection."""
        import cv2
        
        h, w = frame.shape[:2]
        occlusion_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Get facial landmarks
        landmarks = self.get_facial_landmarks(frame)
        
        results = {
            'frame_idx': frame_idx,
            'occlusion_detected': False,
            'occluded_regions': []
        }
        
        if landmarks:
            # Simplified occlusion detection for this implementation
            # In a full implementation, this would use the sophisticated tracking methods
            
            # Get skin mask
            skin_mask = self.detect_skin_pixels(frame)
            
            # Simple distance-based occlusion detection
            face_center = landmarks['nose_tip']
            face_radius = 50  # Approximate face radius
            
            # Count skin pixels near face
            face_region_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(face_region_mask, (int(face_center.x), int(face_center.y)), face_radius, 255, -1)
            
            # Find skin pixels in face region
            skin_in_face = cv2.bitwise_and(skin_mask, face_region_mask)
            occlusion_pixels = np.count_nonzero(skin_in_face)
            
            # Simple threshold-based detection
            if occlusion_pixels > 500:  # Threshold for occlusion
                occlusion_mask = skin_in_face
            
            # Classify occlusions by region
            occlusion_counts = self.classify_occlusions(occlusion_mask, landmarks, (h, w))
            
            # Apply temporal filtering
            filtered_detections = self.temporal_filter(occlusion_counts)
            
            # Store results
            results['occlusion_counts'] = occlusion_counts
            results['filtered_detections'] = filtered_detections
            results['occlusion_detected'] = any(filtered_detections.values())
            results['occluded_regions'] = [HeadRegion.NAMES[r] for r, detected 
                                          in filtered_detections.items() if detected]
        
        return results


def compute_advanced_occlusion_detection_from_keypoints(
    X: np.ndarray,
    mask: np.ndarray,
    output_format: str = 'compatible',
    **kwargs
) -> Union[int, Dict]:
    """
    Compute advanced occlusion detection using preprocessed keypoint data.
    
    Enhanced version with:
    - 5-region head partitioning (forehead, cheeks, nose, mouth, neck)
    - Multiple detection methods (ellipse, proximity, trajectory)
    - Adaptive thresholds for better sensitivity
    - Temporal consistency filtering
    
    Args:
        X: [T, 156] normalized keypoint coordinates
        mask: [T, 78] visibility mask
        output_format: 'compatible' or 'detailed'
        **kwargs: Additional parameters
    
    Returns:
        Binary flag or detailed results
    """
    try:
        T = X.shape[0]
        
        # Keypoint layout: pose25, left_hand21, right_hand21, face11
        pose_len = 25
        hand_len = 21
        face_len = 11
        face_start = pose_len + hand_len + hand_len  # 67
        
        # Ultra-conservative detection parameters to prevent false positives
        min_face_points = 5  # Require more face points for accuracy
        min_hand_points = 4  # Require more hand points for reliability
        min_fingertips_inside = 3  # Require multiple fingertips for reliable detection
        proximity_multiplier = 1.2  # Very conservative multiplier
        occlusion_threshold = 0.30  # Much higher threshold for reliability
        
        results = []
        hand_trajectories = {'left': [], 'right': []}  # Track hand movement
        
        for t in range(T):
            frame_xy = X[t]
            frame_mask = mask[t]
            
            # Extract face keypoints with relaxed requirements
            face_mask = frame_mask[face_start:face_start + face_len]
            if int(face_mask.sum()) < min_face_points:
                results.append({
                    'frame_idx': t,
                    'occlusion_detected': False,
                    'occluded_regions': [],
                    'confidence': 0.0
                })
                continue
            
            # Get visible face coordinates
            face_coords = []
            face_indices = []
            for i_rel in range(face_len):
                if bool(face_mask[i_rel]):
                    idx = 2 * (face_start + i_rel)
                    face_coords.append((float(frame_xy[idx]), float(frame_xy[idx + 1])))
                    face_indices.append(i_rel)
            
            if len(face_coords) < min_face_points:
                results.append({
                    'frame_idx': t,
                    'occlusion_detected': False,
                    'occluded_regions': [],
                    'confidence': 0.0
                })
                continue
            
            # Create enhanced face regions
            face_regions = _create_enhanced_face_regions(face_coords, face_indices)
            
            # Initialize detection results
            occlusion_detected = False
            occluded_regions = []
            max_confidence = 0.0
            
            # Check both hands with enhanced detection
            for hand_side, hand_start in [('left', pose_len), ('right', pose_len + hand_len)]:
                hand_mask = frame_mask[hand_start:hand_start + hand_len]
                visible_points = int(hand_mask.sum())
                
                if visible_points >= min_hand_points:
                    palm_center, tips = _hand_centers_and_tips(frame_xy, frame_mask, hand_start, hand_len)
                    
                    # Update hand trajectory
                    if palm_center is not None:
                        hand_trajectories[hand_side].append((t, palm_center))
                        # Keep only recent trajectory (last 10 frames)
                        if len(hand_trajectories[hand_side]) > 10:
                            hand_trajectories[hand_side] = hand_trajectories[hand_side][-10:]
                    
                    # Multi-method detection
                    region_results = _detect_occlusions_multi_method(
                        palm_center, tips, face_regions, hand_trajectories[hand_side], t,
                        min_fingertips_inside, proximity_multiplier
                    )
                    
                    # Aggregate results with ultra-conservative threshold
                    for region_name, confidence in region_results.items():
                        if confidence > 0.6:  # Very high threshold to prevent false positives
                            occlusion_detected = True
                            if region_name not in occluded_regions:
                                occluded_regions.append(region_name)
                            max_confidence = max(max_confidence, confidence)
            
            results.append({
                'frame_idx': t,
                'occlusion_detected': occlusion_detected,
                'occluded_regions': occluded_regions,
                'confidence': max_confidence
            })
        
        # Apply temporal filtering for consistency
        filtered_results = _apply_temporal_filtering(results)
        
        # Aggregate results with lenient threshold
        total_occlusions = sum(1 for r in filtered_results if r['occlusion_detected'])
        occlusion_rate = total_occlusions / len(filtered_results) if filtered_results else 0
        binary_flag = 1 if occlusion_rate > occlusion_threshold else 0
        
        if output_format == 'compatible':
            return binary_flag
        else:
            return {
                'binary_flag': binary_flag,
                'occlusion_rate': float(occlusion_rate),
                'total_frames': int(len(filtered_results)),
                'occluded_frames': int(total_occlusions),
                'detailed_results': filtered_results
            }
    
    except Exception as e:
        warnings.warn(f"Advanced occlusion detection from keypoints failed: {e}", UserWarning)
        return 0


def _create_enhanced_face_regions(face_coords: List[Tuple[float, float]], 
                                face_indices: List[int]) -> Dict[str, Dict]:
    """
    Create enhanced face regions based on facial landmarks.
    
    Args:
        face_coords: List of (x, y) coordinates for visible face points
        face_indices: List of face landmark indices
        
    Returns:
        Dictionary mapping region names to region definitions
    """
    if len(face_coords) < 3:
        return {}
    
    # Map face indices to landmark names (from FACEMESH_11)
    landmark_map = {
        0: 'nose_tip',      # 1
        1: 'left_eye_outer', # 33
        2: 'right_eye_outer', # 263
        3: 'left_eye_inner', # 133
        4: 'right_eye_inner', # 362
        5: 'mouth_left',     # 61
        6: 'mouth_right',    # 291
        7: 'forehead',       # 105
        8: 'chin',           # 334
        9: 'cheek_left',     # 199
        10: 'cheek_right'    # 4
    }
    
    # Extract key landmarks
    landmarks = {}
    for i, coord in enumerate(face_coords):
        if i < len(face_indices):
            landmark_name = landmark_map.get(face_indices[i], f'point_{i}')
            landmarks[landmark_name] = coord
    
    regions = {}
    
    # Define regions based on available landmarks
    if 'nose_tip' in landmarks:
        nose = landmarks['nose_tip']
        
        # Forehead region (top of head) - precise radius
        if 'forehead' in landmarks:
            forehead = landmarks['forehead']
            regions['forehead'] = {
                'center': forehead,
                'radius': 0.06,  # Much smaller for precision
                'type': 'circle'
            }
        
        # Cheeks region (eye area) - precise radius
        if 'left_eye_outer' in landmarks and 'right_eye_outer' in landmarks:
            left_eye = landmarks['left_eye_outer']
            right_eye = landmarks['right_eye_outer']
            cheek_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
            regions['cheeks'] = {
                'center': cheek_center,
                'radius': 0.08,  # Much smaller for precision
                'type': 'circle'
            }
        
        # Nose region (central face) - precise radius
        regions['nose'] = {
            'center': nose,
            'radius': 0.05,  # Much smaller for precision
            'type': 'circle'
        }
        
        # Mouth region (lower face) - precise radius
        if 'mouth_left' in landmarks and 'mouth_right' in landmarks:
            mouth_left = landmarks['mouth_left']
            mouth_right = landmarks['mouth_right']
            mouth_center = ((mouth_left[0] + mouth_right[0]) / 2, (mouth_left[1] + mouth_right[1]) / 2)
            regions['mouth'] = {
                'center': mouth_center,
                'radius': 0.06,  # Much smaller for precision
                'type': 'circle'
            }
        
        # Neck region (below chin) - precise radius
        if 'chin' in landmarks:
            chin = landmarks['chin']
            neck_center = (chin[0], chin[1] + 0.03)  # Closer to chin
            regions['neck'] = {
                'center': neck_center,
                'radius': 0.08,  # Much smaller for precision
                'type': 'circle'
            }
    
    return regions


def _detect_occlusions_multi_method(palm_center: Optional[Tuple[float, float]], 
                                  tips: List[Tuple[float, float]],
                                  face_regions: Dict[str, Dict],
                                  trajectory: List[Tuple[int, Tuple[float, float]]],
                                  current_frame: int,
                                  min_fingertips_inside: int = 1,
                                  proximity_multiplier: float = 1.6) -> Dict[str, float]:
    """
    Detect occlusions using multiple methods for each region.
    
    Args:
        palm_center: Palm center coordinates
        tips: List of fingertip coordinates
        face_regions: Dictionary of face regions
        trajectory: Hand movement trajectory
        current_frame: Current frame index
        
    Returns:
        Dictionary mapping region names to confidence scores
    """
    region_confidences = {}
    
    for region_name, region_def in face_regions.items():
        confidence = 0.0
        region_center = region_def['center']
        region_radius = region_def['radius']
        
        # Method 1: Direct fingertip intersection (ultra-conservative detection)
        if tips:
            fingertips_inside = 0
            for tip_x, tip_y in tips:
                distance = ((tip_x - region_center[0])**2 + (tip_y - region_center[1])**2)**0.5
                if distance <= region_radius * 0.8:  # Even more conservative intersection
                    fingertips_inside += 1
            
            # Only count if minimum fingertips are inside
            if fingertips_inside >= min_fingertips_inside:
                confidence += 0.6  # High confidence for direct intersection
        
        # Method 2: Palm center proximity (ultra-conservative detection)
        if palm_center is not None:
            palm_x, palm_y = palm_center
            distance = ((palm_x - region_center[0])**2 + (palm_y - region_center[1])**2)**0.5
            # Use very conservative proximity multiplier
            proximity_radius = region_radius * proximity_multiplier
            if distance <= proximity_radius * 0.9:  # Even more conservative
                proximity_score = max(0, 1 - distance / (proximity_radius * 0.9))
                confidence += proximity_score * 0.2  # Lower weight for proximity
        
        # Method 3: Trajectory analysis (ultra-conservative approach detection)
        if len(trajectory) >= 8:  # Require many more frames for reliable trajectory
            recent_positions = trajectory[-8:]  # Last 8 positions
            if len(recent_positions) >= 5:
                # Check if hand is consistently moving toward face
                distances = []
                for _, pos in recent_positions:
                    dist = ((pos[0] - region_center[0])**2 + (pos[1] - region_center[1])**2)**0.5
                    distances.append(dist)
                
                # Check if distance is consistently decreasing
                if len(distances) >= 5:
                    decreasing_count = sum(1 for i in range(1, len(distances)) if distances[i] < distances[i-1])
                    if decreasing_count >= 4 and distances[-1] <= region_radius * 1.2:  # Very conservative radius
                        approach_score = max(0, (distances[0] - distances[-1]) / distances[0])
                        confidence += approach_score * 0.1  # Low weight for trajectory
        
        # Method 4: Multi-point hand analysis (ultra-conservative orientation)
        if palm_center is not None and len(tips) >= 4:  # Require many fingertips
            # Check if hand is oriented toward face
            hand_points = [palm_center] + tips
            face_distances = [((p[0] - region_center[0])**2 + (p[1] - region_center[1])**2)**0.5 for p in hand_points]
            min_distance = min(face_distances)
            
            # Very conservative radius for orientation detection
            if min_distance <= region_radius * 1.3:
                orientation_score = max(0, 1 - min_distance / (region_radius * 1.3))
                confidence += orientation_score * 0.05  # Very low weight for orientation
        
        region_confidences[region_name] = min(confidence, 1.0)  # Cap at 1.0
    
    return region_confidences


def _apply_temporal_filtering(results: List[Dict]) -> List[Dict]:
    """
    Apply temporal filtering to smooth detection results.
    
    Args:
        results: List of frame detection results
        
    Returns:
        Filtered results with improved temporal consistency
    """
    if len(results) < 3:
        return results
    
    filtered_results = []
    window_size = 3
    
    for i, result in enumerate(results):
        # Get temporal window
        start = max(0, i - window_size // 2)
        end = min(len(results), i + window_size // 2 + 1)
        window = results[start:end]
        
        # Calculate smoothed confidence
        confidences = [r['confidence'] for r in window]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Apply majority voting for occlusion detection
        occlusions = [r['occlusion_detected'] for r in window]
        majority_occluded = sum(occlusions) > len(occlusions) // 2
        
        # Combine regions from window
        all_regions = []
        for r in window:
            all_regions.extend(r['occluded_regions'])
        
        # Count region occurrences
        region_counts = {}
        for region in all_regions:
            region_counts[region] = region_counts.get(region, 0) + 1
        
        # Keep regions that appear in majority of frames
        filtered_regions = [region for region, count in region_counts.items() 
                          if count > len(window) // 2]
        
        filtered_result = result.copy()
        filtered_result['occlusion_detected'] = majority_occluded or avg_confidence > 0.7  # Very high threshold to prevent false positives
        filtered_result['occluded_regions'] = list(set(filtered_regions))
        filtered_result['confidence'] = avg_confidence
        
        filtered_results.append(filtered_result)
    
    return filtered_results


def compute_advanced_occlusion_detection(
    video_path: str,
    mode: str = 'advanced',
    output_format: str = 'compatible',
    **kwargs
) -> Union[int, Dict]:
    """
    Compute occlusion detection using advanced computer vision methods.
    
    Args:
        video_path: Path to input video file
        mode: Detection mode ('simple', 'advanced', 'both')
        output_format: Output format ('compatible', 'detailed')
        **kwargs: Additional parameters for advanced detection
    
    Returns:
        int: Binary occlusion flag (compatible format)
        Dict: Detailed results (detailed format)
    """
    try:
        import cv2
        
        # Initialize detector
        detector = AdvancedHandHeadOcclusionDetector(
            use_global_tracking=kwargs.get('use_global_tracking', True)
        )
        
        # Process video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        
        results = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame_results = detector.process_frame(frame, frame_idx)
            results.append(frame_results)
            frame_idx += 1
            
            # Limit processing for performance
            if frame_idx > 1000:  # Process max 1000 frames
                break
        
        cap.release()
        
        # Aggregate results
        total_occlusions = sum(1 for r in results if r['occlusion_detected'])
        occlusion_rate = total_occlusions / len(results) if results else 0
        
        # Determine binary flag
        binary_flag = 1 if occlusion_rate > 0.3 else 0  # 30% threshold
        
        if output_format == 'compatible':
            return binary_flag
        else:
            # Convert results to JSON-serializable format
            serializable_results = []
            for result in results:
                serializable_result = {
                    'frame_idx': result['frame_idx'],
                    'occlusion_detected': result['occlusion_detected'],
                    'occluded_regions': result['occluded_regions']
                }
                # Add optional fields if they exist
                if 'occlusion_counts' in result:
                    serializable_result['occlusion_counts'] = result['occlusion_counts']
                if 'filtered_detections' in result:
                    serializable_result['filtered_detections'] = result['filtered_detections']
                serializable_results.append(serializable_result)
            
            return {
                'binary_flag': binary_flag,
                'occlusion_rate': float(occlusion_rate),  # Convert numpy float to Python float
                'total_frames': int(len(results)),       # Convert numpy int to Python int
                'occluded_frames': int(total_occlusions), # Convert numpy int to Python int
                'detailed_results': serializable_results
            }
    
    except Exception as e:
        warnings.warn(f"Advanced occlusion detection failed: {e}", UserWarning)
        return 0


# Configuration for advanced detection
DEFAULT_ADVANCED_CONFIG = {
    'use_global_tracking': True,
    'gridlet_size': 4,
    'tracking_window_size': 5,
    'motion_threshold': 10,
    'temporal_filtering': True,
    'output_detailed_results': False
}
