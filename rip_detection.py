"""
Rip Current Detection System using OpenCV
==========================================
This program analyzes beach video footage to detect rip currents by identifying
patterns of low foam density in the surf zone. It uses temporal accumulation to
track persistent foam patterns and detect anomalies that indicate rip currents.

Key Concepts:
- Foam Detection: Bright pixels in grayscale indicate breaking waves/foam
- Surf Zone: Area where waves consistently break (high foam activity)
- Rip Currents: Channels of low foam activity perpendicular to shore
- Temporal Accumulation: Builds up patterns over time to reduce false positives
"""

import cv2
import numpy as np

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Video input and processing settings
VIDEO = "test_videos/RipVIS-027.mp4"
RESIZE_WIDTH = 900          # Width to resize video frames (maintains aspect ratio)
FRAME_SKIP = 2              # Process every Nth frame to improve performance

# Foam detection parameters
THRESH = 150                # Initial grayscale threshold for foam detection (auto-calibrated)
DECAY = 0.999999999         # Temporal decay rate for foam accumulation (higher = longer memory)

# Surf zone detection
MASK_THRESH = 180           # Threshold for identifying persistent surf zone
MIN_AREA_RATIO = 0.03       # Minimum contour area as ratio of frame size

# Rip current detection
RIP_DECAY = 0.99            # Persistence decay for rip current heat map
MOVE_THRESH = 6.0           # Pixel movement threshold to detect camera motion

# Surf zone persistence
SURF_DECAY = 0.98           # How quickly surf zone boundary adapts to changes


# ============================================================================
# CAMERA MOTION DETECTION
# ============================================================================

def detect_camera_motion(prev_gray, gray, move_thresh=MOVE_THRESH):
    """
    Detect camera movement between consecutive frames using ORB feature matching.
    
    Camera movement can cause false rip current detections because the entire
    scene shifts. This function uses feature matching to estimate the affine
    transformation between frames.
    
    Args:
        prev_gray: Previous frame in grayscale
        gray: Current frame in grayscale
        move_thresh: Movement threshold in pixels
        
    Returns:
        tuple: (is_moving, movement_magnitude)
            - is_moving: True if camera moved beyond threshold
            - movement_magnitude: Pixel distance of movement
    """
    # Create ORB (Oriented FAST and Rotated BRIEF) feature detector
    orb = cv2.ORB_create(500)
    
    # Detect keypoints and compute descriptors for both frames
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(gray, None)
    
    # Check if features were found in both frames
    if des1 is None or des2 is None:
        return False, None
    
    # Match features using brute-force matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Need minimum matches to estimate transformation reliably
    if len(matches) < 10:
        return False, None
    
    # Extract matched point coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Estimate affine transformation (rotation, scale, translation)
    M, mask = cv2.estimateAffinePartial2D(pts1, pts2)
    
    if M is None:
        return False, None
    
    # Extract translation components (dx, dy) from transformation matrix
    dx, dy = M[0, 2], M[1, 2]
    movement = np.hypot(dx, dy)  # Euclidean distance
    
    return movement > move_thresh, movement


# ============================================================================
# AUTOMATIC THRESHOLD CALIBRATION
# ============================================================================

def auto_calibrate_threshold(cap, sample_frames=50, percentile=85):
    """
    Automatically calibrate foam detection threshold based on video brightness.
    
    Different videos have varying lighting conditions. This function samples
    frames to determine an appropriate threshold that captures foam while
    ignoring darker water.
    
    Args:
        cap: OpenCV VideoCapture object
        sample_frames: Number of frames to sample
        percentile: Percentile of brightness to use as threshold
        
    Returns:
        int: Calibrated threshold value
    """
    vals = []
    pixels_per_frame = 1000  # Sample 1000 pixels per frame for speed
    
    # Sample frames and collect randomly sampled pixel intensity values
    for i in range(sample_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale and flatten to 1D array
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flattened = gray.flatten()
        
        # Randomly sample pixels instead of using all of them
        # This gives ~50x speedup while maintaining accuracy
        if len(flattened) > pixels_per_frame:
            sample_indices = np.random.choice(len(flattened), pixels_per_frame, replace=False)
            sample_pixels = flattened[sample_indices]
        else:
            sample_pixels = flattened
        
        vals.extend(sample_pixels)
    
    # Rewind video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Use percentile to find threshold (85th percentile captures bright foam)
    threshold = np.percentile(vals, percentile)
    
    return int(threshold)


def reset_accumulators():
    """
    Reset all temporal accumulation buffers.
    
    Called when video loops or when significant camera movement is detected
    to prevent accumulating invalid historical data.
    
    Returns:
        tuple: (foam_accumulator, mask_accumulator, rip_accumulator)
    """
    return (
        np.zeros((h, w), np.float32),  # Foam density accumulator
        np.zeros((h, w), np.uint8),     # Mask accumulator (unused currently)
        np.zeros((h, w), np.float32)    # Rip current persistence accumulator
    )


def create_info_panel(width, height, frame_num, fps, thresh, rip_detected, camera_moved):
    """
    Create an information panel showing system status and controls.
    
    Args:
        width: Panel width in pixels
        height: Panel height in pixels
        frame_num: Current frame number
        fps: Current processing FPS
        thresh: Current foam detection threshold
        rip_detected: Boolean indicating if rip is currently detected
        camera_moved: Boolean indicating if camera movement was detected
        
    Returns:
        numpy.ndarray: RGB image of the info panel
    """
    # Create dark gray background
    panel = np.ones((height, width, 3), dtype=np.uint8) * 30
    
    # Define colors (BGR format)
    COLOR_WHITE = (255, 255, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_CYAN = (255, 255, 0)
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    line_height = 22
    
    y_pos = 30
    
    # Title
    cv2.putText(panel, "RIP CURRENT", (10, y_pos), 
                font, 0.65, COLOR_CYAN, 2)
    y_pos += 25
    cv2.putText(panel, "DETECTION SYSTEM", (10, y_pos), 
                font, 0.65, COLOR_CYAN, 2)
    y_pos += 15
    
    # Separator line
    cv2.line(panel, (10, y_pos), (width - 10, y_pos), COLOR_WHITE, 1)
    y_pos += 20
    
    # System Status
    cv2.putText(panel, "STATUS:", (10, y_pos), 
                font, 0.55, COLOR_WHITE, 2)
    y_pos += line_height + 3
    
    # Frame info
    cv2.putText(panel, f"Frame: {frame_num}", (15, y_pos), 
                font, font_scale, COLOR_WHITE, thickness)
    y_pos += line_height
    
    # FPS
    cv2.putText(panel, f"FPS: {fps:.1f}", (15, y_pos), 
                font, font_scale, COLOR_WHITE, thickness)
    y_pos += line_height
    
    # Threshold
    cv2.putText(panel, f"Threshold: {thresh}", (15, y_pos), 
                font, font_scale, COLOR_WHITE, thickness)
    y_pos += line_height + 5
    
    # Rip detection status
    rip_status = "DETECTED!" if rip_detected else "None"
    rip_color = COLOR_RED if rip_detected else COLOR_GREEN
    cv2.putText(panel, "Rip Current:", (15, y_pos), 
                font, font_scale, COLOR_WHITE, thickness)
    y_pos += line_height
    cv2.putText(panel, rip_status, (15, y_pos), 
                font, 0.5, rip_color, 2)
    y_pos += line_height + 5
    
    # Camera movement status
    cv2.putText(panel, "Camera:", (15, y_pos), 
                font, font_scale, COLOR_WHITE, thickness)
    y_pos += line_height
    if camera_moved:
        cv2.putText(panel, "MOVING", (15, y_pos), 
                    font, 0.5, COLOR_YELLOW, 2)
    else:
        cv2.putText(panel, "Stable", (15, y_pos), 
                    font, font_scale, COLOR_GREEN, thickness)
    y_pos += line_height + 10
    
    # Separator line
    cv2.line(panel, (10, y_pos), (width - 10, y_pos), COLOR_WHITE, 1)
    y_pos += 20
    
    # Controls section
    cv2.putText(panel, "CONTROLS:", (10, y_pos), 
                font, 0.55, COLOR_WHITE, 2)
    y_pos += line_height + 3
    
    controls = [
        ("Q", "Quit"),
        ("SPACE", "Pause"),
        ("+", "Inc. thresh"),
        ("-", "Dec. thresh"),
    ]
    
    for key, description in controls:
        cv2.putText(panel, f"{key}:", (15, y_pos), 
                    font, font_scale, COLOR_CYAN, thickness)
        cv2.putText(panel, description, (90, y_pos), 
                    font, font_scale, COLOR_WHITE, thickness)
        y_pos += line_height
    
    y_pos += 10
    
    # Separator line
    cv2.line(panel, (10, y_pos), (width - 10, y_pos), COLOR_WHITE, 1)
    y_pos += 20
    
    # Legend section
    cv2.putText(panel, "LEGEND:", (10, y_pos), 
                font, 0.55, COLOR_WHITE, 2)
    y_pos += line_height + 3
    
    # Green box for surf zone
    cv2.rectangle(panel, (15, y_pos - 12), (35, y_pos - 2), (0, 255, 0), -1)
    cv2.putText(panel, "Surf Zone", (45, y_pos), 
                font, font_scale, COLOR_WHITE, thickness)
    y_pos += line_height
    
    # Red box for rip current
    cv2.rectangle(panel, (15, y_pos - 12), (35, y_pos - 2), (0, 0, 255), -1)
    cv2.putText(panel, "Rip Current", (45, y_pos), 
                font, font_scale, COLOR_WHITE, thickness)
    y_pos += line_height + 5
    
    # Heatmap indicator
    cv2.putText(panel, "Heatmap:", (15, y_pos), 
                font, font_scale, COLOR_WHITE, thickness)
    y_pos += line_height
    cv2.putText(panel, "Blue->Red", (15, y_pos), 
                font, font_scale, COLOR_WHITE, thickness)
    y_pos += line_height
    cv2.putText(panel, "Low->High", (15, y_pos), 
                font, font_scale, COLOR_WHITE, thickness)
    
    return panel


def create_view_descriptions(width, height):
    """
    Create a panel describing each view in the grid.
    
    Args:
        width: Panel width in pixels
        height: Panel height in pixels
        
    Returns:
        numpy.ndarray: RGB image of the descriptions panel
    """
    # Create dark gray background
    panel = np.ones((height, width, 3), dtype=np.uint8) * 30
    
    # Define colors (BGR format)
    COLOR_WHITE = (255, 255, 255)
    COLOR_CYAN = (255, 255, 0)
    COLOR_ORANGE = (0, 165, 255)
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    line_height = 22
    
    y_pos = 30
    
    # Title
    cv2.putText(panel, "VIEW GUIDE", (10, y_pos), 
                font, 0.65, COLOR_CYAN, 2)
    y_pos += 15
    
    # Separator line
    cv2.line(panel, (10, y_pos), (width - 10, y_pos), COLOR_WHITE, 1)
    y_pos += 20
    
    # View descriptions with numbers
    views = [
        ("1", "Original Video", "Raw input feed"),
        ("2", "Foam Heatmap", "Temporal foam"),
        ("3", "Surf Zone", "Breaking waves"),
        ("4", "Combined View", "Overlay blend"),
        ("5", "Rip Detection", "Current frame"),
        ("6", "Rip Heatmap", "Persistent rips"),
        ("7", "Final Output", "Annotated video"),
    ]
    
    for num, title, desc in views:
        # Draw number circle
        cv2.circle(panel, (25, y_pos - 6), 12, COLOR_ORANGE, -1)
        cv2.putText(panel, num, (20, y_pos), 
                    font, 0.5, (0, 0, 0), 2)
        
        # Draw title and description
        cv2.putText(panel, title, (45, y_pos), 
                    font, 0.5, COLOR_WHITE, 2)
        y_pos += line_height - 2
        cv2.putText(panel, desc, (45, y_pos), 
                    font, 0.38, (180, 180, 180), 1)
        y_pos += line_height + 3
    
    return panel


# ============================================================================
# VIDEO INITIALIZATION
# ============================================================================

# Open video file
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit("❌ Video not found or cannot be opened")

# Read first frame to get dimensions
ret, frame = cap.read()
aspect = frame.shape[0] / frame.shape[1]
frame = cv2.resize(frame, (RESIZE_WIDTH, int(RESIZE_WIDTH * aspect)))
h, w = frame.shape[:2]

# Auto-calibrate threshold based on video characteristics
THRESH = auto_calibrate_threshold(cap)
print(f"🎚️  Auto-calibrated foam threshold: {THRESH}")

# Create helper function that doesn't depend on h, w being defined
def reset_accumulators_sized(height, width):
    """Reset accumulators with specific dimensions."""
    return (
        np.zeros((height, width), np.float32),
        np.zeros((height, width), np.uint8),
        np.zeros((height, width), np.float32)
    )

# Initialize temporal accumulation buffers
acc_foam, acc_mask, acc_rip = reset_accumulators_sized(h, w)

# Initialize surf zone accumulator (done separately to avoid NameError later)
surf_acc = np.zeros((h, w), np.float32)

# Previous frame storage for motion detection
prev_gray = None

# FPS calculation variables
frame_count = 0
start_time = cv2.getTickCount()
current_fps = 0.0
camera_moved_flag = False


# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

print("▶️  Starting rip current detection...")
print("Controls: Q=quit, SPACE=pause, +/- adjust threshold")

while True:
    # ========================================================================
    # FRAME ACQUISITION
    # ========================================================================
    
    ret, frame = cap.read()
    
    # Handle end of video by looping
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        acc_foam, acc_mask, acc_rip = reset_accumulators_sized(h, w)
        surf_acc = np.zeros((h, w), np.float32)
        prev_gray = None
        frame_count = 0
        start_time = cv2.getTickCount()
        print("🔁 Looping video, accumulators reset")
        continue
    
    # Skip frames for performance
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % FRAME_SKIP != 0:
        continue
    
    # Update FPS counter
    frame_count += 1
    if frame_count % 30 == 0:  # Update every 30 frames
        elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        current_fps = frame_count / elapsed if elapsed > 0 else 0.0
    
    # Resize and convert to grayscale
    frame = cv2.resize(frame, (RESIZE_WIDTH, int(RESIZE_WIDTH * aspect)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ========================================================================
    # CAMERA MOTION DETECTION
    # ========================================================================
    
    camera_moved_flag = False
    if prev_gray is not None:
        moved, motion_mag = detect_camera_motion(prev_gray, gray)
        
        if moved:
            camera_moved_flag = True
            print(f"📷 Camera movement detected: {motion_mag:.2f}px - resetting accumulators")
            # Reset accumulators to prevent false detections from camera movement
            acc_foam *= 0
            acc_rip *= 0
            surf_acc *= 0
    
    prev_gray = gray.copy()
    
    # ========================================================================
    # FOAM DETECTION
    # ========================================================================
    
    # Threshold to binary: bright pixels (foam) = 255, dark pixels (water) = 0
    _, foam = cv2.threshold(gray, THRESH, 255, cv2.THRESH_BINARY)
    
    # Temporal accumulation: blend current foam with historical foam
    # Higher DECAY means longer memory of past foam locations
    acc_foam = acc_foam * DECAY + (foam / 255.0) * (1 - DECAY)
    
    # Normalize accumulated foam to 0-255 range for visualization
    foam_density = cv2.normalize(acc_foam, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Create heatmap visualization (blue=low, red=high foam density)
    foam_heat = cv2.applyColorMap(foam_density, cv2.COLORMAP_JET)
    
    # ========================================================================
    # SURF ZONE DETECTION
    # ========================================================================
    
    # Find contours of current foam regions
    contours, _ = cv2.findContours(foam, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for large contours (ignore small foam patches)
    MIN_AREA = MIN_AREA_RATIO * (h * w)
    large_contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]
    
    # Create convex hull around all large foam regions
    frame_mask = np.zeros((h, w), np.uint8)
    if large_contours:
        # Combine all contours into one point set
        all_points = np.vstack(large_contours)
        hull = cv2.convexHull(all_points)
        cv2.drawContours(frame_mask, [hull], -1, 255, -1)
    
    # Temporally accumulate surf zone (slower decay than foam)
    surf_acc = surf_acc * SURF_DECAY + (frame_mask / 255.0) * (1 - SURF_DECAY)
    
    # Threshold to get persistent surf zone boundary
    _, surf_mask = cv2.threshold(surf_acc, 0.3, 1.0, cv2.THRESH_BINARY)
    surf_mask = (surf_mask * 255).astype(np.uint8)
    
    # Smooth surf zone boundary for visualization
    contours, _ = cv2.findContours(surf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = frame.copy()
    
    if contours:
        # Create smooth polygon around surf zone
        all_points = np.vstack(contours)
        hull = cv2.convexHull(all_points)
        epsilon = 0.005 * cv2.arcLength(hull, True)
        poly = cv2.approxPolyDP(hull, epsilon, True)
        
        # Draw green boundary and semi-transparent fill
        cv2.drawContours(overlay, [poly], -1, (0, 255, 0), 2)
        cv2.fillPoly(overlay, [poly], (0, 255, 0))
    
    surf_zone = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    
    # ========================================================================
    # RIP CURRENT DETECTION
    # ========================================================================
    
    # Only look for rips within the surf zone
    foam_in_zone = cv2.bitwise_and(foam_density, foam_density, mask=surf_mask)
    
    # Calculate adaptive threshold based on average foam in surf zone
    zone_values = foam_in_zone[surf_mask == 255]
    
    if len(zone_values) > 0:
        mean_val = np.mean(zone_values)
        # Rip channels have lower foam (60% of average)
        rip_thresh = max(20, mean_val * 0.6)
    else:
        rip_thresh = 80
    
    # Invert threshold: find areas with LOW foam (potential rips)
    _, lowfoam = cv2.threshold(foam_in_zone, rip_thresh, 255, cv2.THRESH_BINARY_INV)
    
    # Clean up noise with morphological opening
    lowfoam = cv2.morphologyEx(lowfoam, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    
    # Ensure we only consider areas within surf zone
    lowfoam = cv2.bitwise_and(lowfoam, surf_mask)
    
    # Find rip current candidates
    contours, _ = cv2.findContours(lowfoam, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rip_overlay = frame.copy()
    rip_mask = np.zeros((h, w), np.uint8)
    
    if contours:
        # Use largest low-foam region as rip candidate
        largest = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest) > 0.001 * (h * w):
            # Smooth the rip boundary using Gaussian blur
            rip_mask_tmp = np.zeros((h, w), np.uint8)
            cv2.drawContours(rip_mask_tmp, [largest], -1, 255, -1)
            rip_mask_tmp = cv2.GaussianBlur(rip_mask_tmp, (11, 11), 0)
            _, rip_mask_tmp = cv2.threshold(rip_mask_tmp, 127, 255, cv2.THRESH_BINARY)
            
            # Find smoothed contour
            contours_smooth, _ = cv2.findContours(rip_mask_tmp, cv2.RETR_EXTERNAL, 
                                                   cv2.CHAIN_APPROX_SIMPLE)
            
            if contours_smooth:
                smoothed = max(contours_smooth, key=cv2.contourArea)
                epsilon = 0.005 * cv2.arcLength(smoothed, True)
                smooth_poly = cv2.approxPolyDP(smoothed, epsilon, True)
                
                # Draw red rip current region
                cv2.drawContours(rip_overlay, [smooth_poly], -1, (0, 0, 255), 2)
                cv2.fillPoly(rip_overlay, [smooth_poly], (0, 0, 255))
                cv2.drawContours(rip_mask, [smooth_poly], -1, 255, -1)
                
                rip_overlay = cv2.addWeighted(rip_overlay, 0.4, frame, 0.6, 0)
    
    # ========================================================================
    # RIP PERSISTENCE HEATMAP
    # ========================================================================
    
    # Accumulate rip detections over time to show persistent rips
    acc_rip = acc_rip * RIP_DECAY + (rip_mask / 255.0) * (1 - RIP_DECAY)
    
    # Create heatmap showing areas with persistent low foam
    rip_heat = cv2.normalize(acc_rip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    rip_heat = cv2.applyColorMap(rip_heat, cv2.COLORMAP_JET)
    
    # ========================================================================
    # FINAL RIP ANNOTATION (from persistence heatmap)
    # ========================================================================
    
    # Threshold the persistence map to find strong rip signals
    heat_bin = cv2.normalize(acc_rip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, heat_mask = cv2.threshold(heat_bin, 100, 255, cv2.THRESH_BINARY)
    heat_mask = cv2.bitwise_and(heat_mask, heat_mask, mask=surf_mask)
    
    # Find contours of persistent rips
    contours_h, _ = cv2.findContours(heat_mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
    heat_poly = None
    
    if contours_h:
        largest_h = max(contours_h, key=cv2.contourArea)
        
        if cv2.contourArea(largest_h) > 0.001 * (h * w):
            eps_h = 0.005 * cv2.arcLength(largest_h, True)
            heat_poly = cv2.approxPolyDP(largest_h, eps_h, True)
    
    # Create final annotated frame with detected rip
    rip_annot = frame.copy()
    rip_detected = False
    
    if heat_poly is not None:
        rip_detected = True
        # Draw red rip current warning on original video
        cv2.polylines(rip_annot, [heat_poly], isClosed=True, 
                     color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
        cv2.fillPoly(rip_annot, [heat_poly], (0, 0, 255))
        rip_annot = cv2.addWeighted(rip_annot, 0.35, frame, 0.65, 0)
    
    # ========================================================================
    # CREATE INFO PANEL
    # ========================================================================
    
    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    info_panel = create_info_panel(300, frame.shape[0], frame_num, 
                                   current_fps, THRESH, rip_detected, camera_moved_flag)
    
    # Create view descriptions panel
    view_desc_panel = create_view_descriptions(300, frame.shape[0])
    
    # ========================================================================
    # VISUALIZATION GRID
    # ========================================================================
    
    # Create combined overlay (surf zone + foam heatmap)
    combined = cv2.addWeighted(surf_zone, 0.5, foam_heat, 0.5, 0)
    
    # ========================================================================
    # ADD NUMBERED LABELS TO EACH VIEW
    # ========================================================================
    
    def add_label(img, number, label):
        """Add a numbered label to the top-left corner of an image."""
        labeled = img.copy()
        # Draw semi-transparent background box
        cv2.rectangle(labeled, (5, 5), (170, 35), (0, 0, 0), -1)
        cv2.rectangle(labeled, (5, 5), (170, 35), (255, 255, 255), 2)
        # Draw number circle
        cv2.circle(labeled, (20, 20), 12, (0, 165, 255), -1)
        cv2.putText(labeled, str(number), (15, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        # Draw label text
        cv2.putText(labeled, label, (38, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return labeled
    
    # Label all views
    frame_labeled = add_label(frame, 1, "Original")
    foam_heat_labeled = add_label(foam_heat, 2, "Foam Heatmap")
    surf_zone_labeled = add_label(surf_zone, 3, "Surf Zone")
    combined_labeled = add_label(combined, 4, "Combined")
    rip_overlay_labeled = add_label(rip_overlay, 5, "Rip Detection")
    rip_heat_labeled = add_label(rip_heat, 6, "Rip Heatmap")
    rip_annot_labeled = add_label(rip_annot, 7, "Final Output")
    
    # Create 2-row grid of visualizations
    # Top row: Original | Foam Heatmap | Surf Zone | Info Panel | View Descriptions
    grid_top = np.hstack([frame_labeled, foam_heat_labeled, surf_zone_labeled, 
                          info_panel, view_desc_panel])
    
    # Bottom row: Combined | Rip Detection | Rip Heatmap | Final Annotation
    grid_bottom = np.hstack([combined_labeled, rip_overlay_labeled, 
                            rip_heat_labeled, rip_annot_labeled])
    
    # Ensure both rows have same width for stacking
    max_width = max(grid_top.shape[1], grid_bottom.shape[1])
    
    if grid_bottom.shape[1] < max_width:
        pad = np.zeros((grid_bottom.shape[0], max_width - grid_bottom.shape[1], 3), 
                      np.uint8)
        grid_bottom = np.hstack([grid_bottom, pad])
    elif grid_top.shape[1] < max_width:
        pad = np.zeros((grid_top.shape[0], max_width - grid_top.shape[1], 3), 
                      np.uint8)
        grid_top = np.hstack([grid_top, pad])
    
    # Stack rows vertically
    grid = np.vstack([grid_top, grid_bottom])
    
    # Display complete visualization
    cv2.imshow("Rip Current Detection System", grid)
    
    # ========================================================================
    # KEYBOARD CONTROLS
    # ========================================================================
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        # Quit program
        break
    elif key == ord(' '):
        # Pause/unpause
        print("⏸️  Paused. Press any key to continue...")
        cv2.waitKey(0)
    elif key == ord('+'):
        # Increase threshold (detect less foam)
        THRESH = min(255, THRESH + 5)
        print(f"↑ Threshold: {THRESH}")
    elif key == ord('-'):
        # Decrease threshold (detect more foam)
        THRESH = max(0, THRESH - 5)
        print(f"↓ Threshold: {THRESH}")

# ============================================================================
# CLEANUP
# ============================================================================

cap.release()
cv2.destroyAllWindows()
print("✅ Program terminated successfully")