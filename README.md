# Rip Current Detection System

A computer vision system that analyzes beach video footage to detect rip currents using OpenCV. The system identifies patterns of low foam density in the surf zone and uses temporal accumulation to track persistent foam patterns and detect anomalies that indicate dangerous rip currents.

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌊 What Are Rip Currents?

Rip currents are powerful, narrow channels of fast-moving water that flow away from the shore. They appear as areas of reduced wave breaking (less foam) and can be deadly to swimmers. This system helps detect these currents automatically by analyzing video footage.

## ⚠️ Disclaimer

**This is a proof-of-concept implementation** designed to demonstrate the potential of image processing techniques with OpenCV for rip current detection. This version has many possible improvements and should **NOT** be used as the sole method for beach safety assessment.

**Limitations:**
- Not suitable for real-time safety monitoring without extensive validation
- Detection accuracy varies significantly based on lighting, weather, and camera conditions
- Requires manual threshold adjustments for optimal performance in some scenarios
- False positives can occur due to camera movement, shadows, or unusual wave patterns
- Should only be used as a supplementary tool alongside professional lifeguard assessment

**Future improvements could include:**
- Machine learning-based detection for better accuracy
- Multi-camera fusion for wider coverage
- Real-time alert systems
- Better handling of variable lighting conditions
- Integration with weather and tide data

## 🎯 How It Works

The system uses several computer vision techniques:

1. **Foam Detection**: Identifies bright pixels in grayscale video as breaking waves/foam
2. **Temporal Accumulation**: Builds up foam patterns over time to reduce false positives
3. **Surf Zone Identification**: Detects the area where waves consistently break
4. **Rip Current Detection**: Identifies channels of persistently low foam activity within the surf zone
5. **Camera Motion Compensation**: Detects and compensates for camera movement to prevent false alarms

## 📋 Requirements

```bash
pip install opencv-python numpy
```

**Minimum Requirements:**
- Python 3.7+
- OpenCV 4.0+
- NumPy 1.19+

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd rip-current-detection
```

### 2. Download Test Videos

Sample videos can be downloaded from the RipVIS dataset:
- Dataset URL: https://huggingface.co/datasets/Irikos/RipVIS/tree/main/test/videos
- Download one or more `.mp4` files to a `test_videos/` directory

### 3. Run the Detection System

```bash
python rip_detection.py
```

The system will:
1. **Auto-calibrate** the foam detection threshold based on the video's brightness
2. Process the video and display a multi-view interface
3. Detect and highlight rip currents in real-time

### 4. Select a Different Video

Edit the `VIDEO` parameter at the top of `rip_detection.py`:

```python
VIDEO = "test_videos/RipVIS-027.mp4"  # Change this to your video file
```

Or pass it as a command-line argument (requires minor code modification).

## 🎮 Controls

| Key | Action | Description |
|-----|--------|-------------|
| **Q** | Quit | Exit the program |
| **SPACE** | Pause | Pause/unpause video playback |
| **+** | Increase Threshold | Detect less foam (use if too much is detected) |
| **-** | Decrease Threshold | Detect more foam (use if too little is detected) |

### Threshold Adjustment

While the system **automatically calibrates** the foam detection threshold at startup, you may need to make small manual adjustments depending on:
- Changing lighting conditions during the video
- Shadows or glare
- Different water/foam characteristics

**Tip:** If the system detects foam everywhere, press `+` a few times. If it's not detecting enough foam in breaking waves, press `-`.

## 📺 Understanding the Display

The interface shows 7 different views arranged in a grid:

### Top Row:
1. **Original Video** - Raw input footage
2. **Foam Heatmap** - Temporal accumulation of foam (blue = low, red = high)
3. **Surf Zone** - Detected breaking wave area (green overlay)
4. **Status Panel** - System information and controls
5. **View Guide** - Description of each view

### Bottom Row:
4. **Combined View** - Overlay of surf zone and foam heatmap
5. **Rip Detection** - Current frame rip detection (red areas)
6. **Rip Heatmap** - Persistent rip current accumulation over time
7. **Final Output** - Annotated video with detected rip currents

### Status Panel Information:
- **Frame**: Current frame number
- **FPS**: Processing frames per second
- **Threshold**: Current foam detection threshold value
- **Rip Current**: Detection status (None/DETECTED!)
- **Camera**: Camera stability status (Stable/MOVING)

### Visual Legend:
- 🟢 **Green Overlay**: Surf zone (area where waves break)
- 🔴 **Red Overlay**: Detected rip current
- 🌡️ **Heatmap Colors**: Blue (low) → Green → Yellow → Red (high)

## 🔧 Configuration Parameters

You can modify these parameters at the top of `rip_detection.py`:

```python
# Video settings
VIDEO = "test_videos/RipVIS-027.mp4"  # Path to video file
RESIZE_WIDTH = 900                     # Display width (maintains aspect ratio)
FRAME_SKIP = 2                         # Process every Nth frame

# Foam detection
THRESH = 150                           # Auto-calibrated at startup
DECAY = 0.999999999                    # Foam memory (higher = longer)

# Surf zone detection
MASK_THRESH = 180                      # Surf zone persistence threshold
MIN_AREA_RATIO = 0.03                  # Minimum contour size (3% of frame)

# Rip current detection
RIP_DECAY = 0.99                       # Rip persistence decay rate
MOVE_THRESH = 6.0                      # Camera movement threshold (pixels)

# Surf zone adaptation
SURF_DECAY = 0.98                      # Surf zone boundary adaptation rate
```

## 📁 Project Structure

```
rip-current-detection/
│
├── rip_detection.py          # Main detection system
├── README.md                  # This file
│
└── test_videos/              # Video files directory
    ├── RipVIS-027.mp4
    └── ...
```

## 🎓 How to Get More Test Videos

The RipVIS dataset contains various beach videos suitable for testing:

1. Visit: https://huggingface.co/datasets/Irikos/RipVIS/tree/main/test/videos
2. Download any `.mp4` files you want to test
3. Place them in the `test_videos/` directory
4. Update the `VIDEO` variable in the code

The dataset includes videos with different:
- Lighting conditions (sunny, cloudy, dawn, dusk)
- Wave characteristics
- Camera angles and distances
- Presence/absence of rip currents

## 🐛 Troubleshooting

### "Video not found or cannot be opened"
- Check that the video file path is correct
- Ensure the video file is in a supported format (MP4, AVI, MOV)
- Try using an absolute path instead of relative path

### Detection is too sensitive / not sensitive enough
- Use `+` and `-` keys to adjust threshold during playback
- Edit the `THRESH` parameter for different default starting point
- Try different videos with better lighting conditions

### Program is running slowly
- Increase `FRAME_SKIP` to process fewer frames (e.g., `FRAME_SKIP = 3`)
- Reduce `RESIZE_WIDTH` for smaller processing window (e.g., `RESIZE_WIDTH = 640`)
- Close other applications to free up CPU resources

### Camera movement keeps resetting detection
- Reduce `MOVE_THRESH` if camera movements are too small to matter
- Use videos from stationary cameras when possible
- The system is designed to reset on movement to prevent false positives

### No rip currents detected in video
- Verify the video actually contains visible rip currents
- Try adjusting the threshold with `+` and `-` keys
- Check that the surf zone (green area) is being detected correctly
- Some videos may not have clear enough rip signatures

## 🔬 Technical Details

### Detection Algorithm

1. **Frame Preprocessing**
   - Resize for consistent processing
   - Convert to grayscale
   - Apply binary threshold to detect bright foam

2. **Temporal Accumulation**
   - Exponential moving average of foam presence
   - Longer memory helps filter out random wave variations

3. **Surf Zone Extraction**
   - Find large contiguous foam regions
   - Create convex hull around all foam areas
   - Accumulate over time for stable boundary

4. **Rip Current Identification**
   - Calculate adaptive threshold based on average foam in surf zone
   - Detect areas with abnormally low foam (< 60% of average)
   - Filter small regions and smooth boundaries
   - Track persistence over multiple frames

5. **Camera Motion Handling**
   - ORB feature detection and matching
   - Estimate affine transformation between frames
   - Reset accumulators if significant motion detected

### Performance Considerations

- Processing speed: ~15-30 FPS on modern hardware (depends on video resolution)
- Memory usage: ~100-200 MB (depends on video size)
- The system is designed for offline analysis, not real-time streaming

## 📝 License

This project is released under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- RipVIS dataset for test videos
- OpenCV community for excellent documentation
- Research on rip current detection using computer vision

## 📧 Contact

For questions, suggestions, or issues, please open an issue on the project repository.

---

**Remember**: This is a demonstration project to showcase the potential of computer vision for coastal safety applications. Always prioritize professional lifeguard assessment and official beach safety warnings.