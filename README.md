# SLAM with Colmap
# **author**: Thanh B. Huynh

## 1. Pipeline Structure
**1. Input**:
1. *Image*: .png, .jpg, .jpeg
2. *Video*: .mp4, .insv
3. *Frame* sequenc + IMU: .lrv, .insv 

**2. Pre-processing**

**3. SLAM**:
1. Feature extraction (colmap)
2. Sequence matching (colmap)
3. Camera pose estimation (glomap)

**4. Trajectory**