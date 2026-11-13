# **SLAM with Colmap**
**Author**: Thanh B. Huynh
---
This small project utilize the `Colmap` SLAM pipeline to process a custom dataset. The dataset is created by using `fisheye lens` camera and store in `.insv` format. The aim is to create a set of camera pose trajectory according to recorded videos.
## I. Install Colmap and Glomap
1. Install `Colmap` and `Glomap` following the official guide:

**Colmap**: [Install](https://colmap.github.io/install.html#build-from-source)[^1]

**Glomap**: [Install](https://github.com/colmap/glomap/tree/main)[^2]

2. Install this package:
```bash
uv sync
```
## II. Pipeline Structure
**1. Input**:
1. *Image*: .png, .jpg, .jpeg
2. *Video*: .mp4, .insv
3. *Frame* sequenc + IMU: .lrv, .insv 

**2. Pre-processing**

**3. SLAM**:
1. Feature extraction (colmap)
2. Sequence matching (colmap)
3. Camera pose estimation (glomap or colmap)

**4. Trajectory** 

## III. Quick start
**1. Pre-processing:**
```bash
cd src/
uv run python -m slam_colmap.pre-processing --input <file-name>.insv --output_dir <Path>
```

**2. SLAM**
* Feature extraction:
```bash
uv run python -m slam_colmap.slam extract --workspace <Path>
```
* Sequential Matching:
```bash
uv run python -m slam_colmap.slam matching --workspace <Path>
```
* Map Construction:
```bash
uv run python -m slam_colmap.slam mapper --workspace <Path>
```

**3. Post-processing:**
```bash
uv run python -m slam_colmap.post-processing --workspace <Path>
```

**4. Visualization:**
```bash
uv run python -m slam_colmap.visualize_trajectory --file <Path>
```

## **IV. Reference**

[^1]: Schönberger, J. L., & Frahm, J.-M. (2016). *Structure-from-Motion Revisited*. In **Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)**.

[^2]: Pan, L., Barath, D., Pollefeys, M., & Schönberger, J. L. (2024). *Global Structure-from-Motion Revisited*. In **Proceedings of the European Conference on Computer Vision (ECCV)**.