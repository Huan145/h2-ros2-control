# FoundationPose Static Pose Estimation — Baseline Milestone

## Files Needed
```
convert_obj_cm_to_m.py
semantic_mask_node.py
foundationpose_klt.launch.py
pose_overlay_visualizer.py
```

## Key File Paths
```
Mesh (original, cm):   /home/apt-ipc/Downloads/KLT_box/KLT_box.obj
Mesh (converted, m):   /home/apt-ipc/Downloads/KLT_box/KLT_box_metres.obj
Refine engine:         /workspaces/isaac_ros-dev/isaac_ros_assets/models/foundationpose/refine_trt_engine.plan
Score engine:          /workspaces/isaac_ros-dev/isaac_ros_assets/models/foundationpose/score_trt_engine.plan
```

---

## One-Time Setup (do this once, not every session)

### Step 1a — Tag the KLT box with a semantic label in Isaac Sim
- Open Isaac Sim and load your scene
- Select the KLT box prim in the Stage panel (`/World/KLT_box_2/KLT_box_1_`)
- Open `Tools → Replicator → Semantics Schema Editor`
- Set `New Semantic Labels` = `klt_box`
- Click `Add Entry On All Selected Prims`
- Save the scene so the tag persists

### Step 1b — Convert mesh from centimetres to metres
```bash
python3 convert_obj_cm_to_m.py
```
Verify the output:
```bash
grep "^v " /home/apt-ipc/Downloads/KLT_box/KLT_box_metres.obj | head -3
# Should show small values like: 0.15 -0.20 0.075
# NOT large values like:         15 -20 7.5
```
> ⚠️ This only needs to be done **once**. `KLT_box_metres.obj` can be reused every session.

### Step 1c — Verify launch file points to converted mesh
Check `foundationpose_klt.launch.py`:
```python
MESH_FILE_PATH = '/home/apt-ipc/Downloads/KLT_box/KLT_box_metres.obj'
```

---

## Every Session

### Step 2 — Find the KLT box semantic ID (changes every Isaac Sim session)
Open Isaac Sim, play the simulation, then run this in the **Script Editor**
(`Window → Script Editor`):
```python
import asyncio
import omni.replicator.core as rep

async def get_mapping():
    annotator = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
    rp = rep.create.render_product("/World/Camera", (1280, 720))
    annotator.attach(rp)
    await rep.orchestrator.step_async()
    data = annotator.get_data()
    mapping = data.get("info", {}).get("idToLabels", {})
    print("=== ID to Label Mapping ===")
    for id, label in mapping.items():
        print(f"  ID {id}: {label}")
    print("===========================")

asyncio.ensure_future(get_mapping())
```
Look for the line with `klt_box` and note the ID number, e.g.:
```
ID 9: {'klt_box_1': 'klt_box', 'class': 'defaultmaterial,mesh'}
```

### Step 3 — Update semantic ID in mask node
Open `semantic_mask_node.py` and update line:
```python
TARGET_SEMANTIC_ID = 9   # ← replace with ID from Step 2
```

### Step 4 — Record a rosbag with the KLT box static and visible
Inside the Docker container:
```bash
source /opt/ros/jazzy/setup.bash
ros2 bag record /rgb /depth /camera_info /semantic_segmentation \
  -o klt_box_static
```
- Keep the KLT box **static and fully visible** in the camera frame
- Record for ~30 seconds
- Press `Ctrl+C` to stop recording

### Step 5 — Close Isaac Sim
Isaac Sim uses a lot of VRAM. Close it before running FoundationPose to avoid
running out of GPU memory (important for RTX 3080 with 10GB VRAM).

---

## Running the Pipeline (5 terminals)

> ⚠️ All terminals must be inside the Docker container with ROS2 sourced.

**Terminal 1 — Semantic mask node:**
```bash
source /opt/ros/jazzy/setup.bash
python3 semantic_mask_node.py
```
Expected output:
```
SemanticMaskNode started. Watching for semantic ID=9 on /semantic_segmentation
Published mask: 14523 / 921600 pixels are KLT box
```

**Terminal 2 — FoundationPose (wait for "Node was started" before proceeding):**
```bash
source /opt/ros/jazzy/setup.bash
ros2 launch foundationpose_klt.launch.py
```
Wait until you see:
```
[foundationpose_node]: [NitrosNode] Node was started
```
> ⚠️ This takes 20-60 seconds to initialize TensorRT engines. Be patient.

**Terminal 3 — Play rosbag:**
```bash
source /opt/ros/jazzy/setup.bash
ros2 bag play ~/klt_box_static --clock --loop
```
> ⚠️ Only start the bag AFTER FoundationPose prints "Node was started"

**Terminal 4 — Overlay visualizer:**
```bash
source /opt/ros/jazzy/setup.bash
python3 pose_overlay_visualizer.py
```
Expected output:
```
PoseOverlayVisualizer ready → /pose_visualization
Pose updated: x=... y=... z=...m
```

**Terminal 5 — RViz2 (on host machine, outside container):**
```bash
source /opt/ros/jazzy/setup.bash
rviz2
```
In RViz2:
- Set **Fixed Frame** to `sim_camera`
- Click **Add → By topic** → select `/pose_visualization` → **Image**

---

## Expected Output

| Topic | Rate | Description |
|---|---|---|
| `/segmentation_mask` | ~10Hz | White blob on black — KLT box masked |
| `/output` | ~1.5Hz | 6-DoF pose as `Detection3DArray` |
| `/pose_visualization` | ~10Hz | RGB image with green wireframe + XYZ axes |

The `/pose_visualization` image should show:
- **Green wireframe box** overlaid on the KLT box in the RGB image
- **Red arrow** = X axis
- **Green arrow** = Y axis
- **Blue arrow** = Z axis
- **Position text** in top-left corner: `x=... y=... z=...m`

---

## Verifying Pose Accuracy

Compare FoundationPose output against Isaac Sim ground truth.

**FoundationPose output:**
```bash
ros2 topic echo /output --field detections[0].results[0].pose.pose.position --once
```

**Isaac Sim ground truth** (run in Script Editor with camera at world origin):
```python
import omni.usd
from pxr import UsdGeom

stage = omni.usd.get_context().get_stage()
box_prim = stage.GetPrimAtPath("/World/KLT_box_2")
box_T = UsdGeom.Xformable(box_prim).ComputeLocalToWorldTransform(0)
box_pos = box_T.ExtractTranslation()
print(f"Box position (m): x={box_pos[0]:.3f} y={box_pos[1]:.3f} z={box_pos[2]:.3f}")
```

> ℹ️ Coordinates won't match directly due to axis convention differences between
> Isaac Sim world frame and camera frame. However the **Euclidean distance**
> (magnitude) should match:
> ```python
> import numpy as np
> isaac = np.array([x, y, z])   # from Isaac Sim
> fp    = np.array([x, y, z])   # from FoundationPose
> print(f"Isaac distance:        {np.linalg.norm(isaac):.3f}m")
> print(f"FoundationPose distance: {np.linalg.norm(fp):.3f}m")
> # These should be approximately equal
> ```

---

## Known Limitations (to be addressed in next milestones)

| Issue | Cause | Planned Fix |
|---|---|---|
| Pose updates at ~1.5Hz | No tracking, full estimation every frame | Add Selector + Tracking node |
| Axes flip 90° occasionally | Box symmetry ambiguity | Add tracking + texture to mesh |
| Semantic ID changes each session | Isaac Sim dynamic ID assignment | Auto-detect ID by label name |
| Not suitable for moving box | Low update rate without tracking | Add tracking pipeline |

---

## Next Milestone
**Milestone 2 — Continuous Tracking**
Add `Selector` + `FoundationPoseTrackingNode` for smooth ~20Hz pose updates
suitable for a moving KLT box.
