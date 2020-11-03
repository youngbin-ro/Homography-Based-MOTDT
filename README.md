# Homography-Based-MOTDT
- Term project results for AAA534 \<Computer Vision\> in Korea University
- This work is based on MOTDT which is one of the state-of-the-art algorithm for real-time multiple object tracking
  - code: https://github.com/longcw/MOTDT
  - paper: https://arxiv.org/abs/1809.04427
- For more information, please refer to the [report](https://github.com/youngbin-ro/Homography-Based-MOTDT/blob/master/report.pdf) file in this repository

<br/>

## Overview
![overview](https://github.com/youngbin-ro/Homography-Based-MOTDT/blob/master/images/overview.PNG?raw=true)

- **STEP1**: Estimate bounding box of frame ```t+1``` from the current frame ```t``` through Kalman Filter
- **STEP2**: Detect object at time ```t+1``` using R-FCN
- **STEP3**: Filter objects estimated in **STEP1** and objects detected in **STEP2** through Non-Maximum Suppression
- **STEP4**: Calculate homography matrix from frame ```t``` and ```t+1```
- **STEP5**: Create candidates by linearly transforming the existing object at time ```t``` through homography matrix obtained in **STEP4**
- **STEP6**: Allocate bounding box candidates from **STEP3** and **STEP5** to each object based on IOU and ReIE features.

<br/>

## Tracking Examples

### MOT17 Dataset
#### MOTDT (original)
<img style="float: left;" src="https://github.com/youngbin-ro/Homography-Based-MOTDT/blob/master/images/MOT17_original.gif?raw=true">

- The original model cannot maintain the track ID of object 1 (turned to 101), which is covered by object 105

#### Homography Based MOTDT (proposed)
<img style="float: left;" src="https://github.com/youngbin-ro/Homography-Based-MOTDT/blob/master/images/MOT17_proposed.gif?raw=true">

- Ours maintains the track ID of object 1 and 89 even though they are obscured by object 161 carrying a green bag.

### VisDrone Dataset
#### MOTDT (original)
<img style="float: left;" src="https://github.com/youngbin-ro/Homography-Based-MOTDT/blob/master/images/VisDrone_original.gif?raw=true">

- The original model cannot maintain the track ID of object 427  (turned to 509) due to a sudden change in camera angle

#### Homography Based MOTDT (proposed)
<img style="float: left;" src="https://github.com/youngbin-ro/Homography-Based-MOTDT/blob/master/images/VisDrone_proposed.gif?raw=true">

- Ours maintains the track ID of object 515 even though there is a sudden change in camera angle at the end of the clip

<br/>

## Results
#### MOT17 Dataset

|                | **Original** | **Proposed** |
| :--------: | :------------------: | :-----: |
|       **idf1** | 0.503 | **<u>0.522</u>** |
| **Mostly Tracked** | 59 | **<u>70</u>** |
| **Mostly Lost** | **<u>151</u>** | 152 |
| **False Positive** | **<u>919</u>** | 3,057 |
| **Num_Misses** | 28,580 | **<u>26,781</u>** |
| **Num_Switches** | 200 | **<u>198</u>** |
| **Num_Fragment** | 706 | **<u>574</u>** |
|       **MOTA** | **<u>0.428</u>** | 0.421 |
|       **MOTP** | 0.152 | **<u>0.164</u>** |

#### VisDrone Dataset

|                | **Original** | **Proposed** |
| :--------: | :------------------: | :-----: |
|       **idf1** | 0.547 | **<u>0.579</u>** |
| **Mostly Tracked** | 75 | **<u>97</u>** |
| **Mostly Lost** | **<u>94</u>** | 97 |
| **False Positive** | **<u>725</u>** | 3,064 |
| **Num_Misses** | 22,704 | **<u>19,818</u>** |
| **Num_Switches** | 504 | **<u>386</u>** |
| **Num_Fragment** | 1,604 | **<u>806</u>** |
|       **MOTA** | 0.524 | **<u>0.538</u>** |
|       **MOTP** | 0.094 | **<u>0.116</u>** |















