AirSLAM: An Efficient and Illumination-Robust Point-Line Visual SLAM System

Abstract - In this paper, we present an efficient visual SLAM system designed to tackle both short-term and long-term illumination challenges. Our system adopts a hybrid approach that combines deep learning techniques for feature detection and matching with traditional backend optimization methods. Specifically, we propose a unified convolutional neural network (CNN) that simultaneously extracts keypoints and structural lines. These features are then associated, matched, triangulated, and optimized in a coupled manner. Additionally, we introduce a lightweight relocalization pipeline that reuses the built map, where keypoints, lines, and a structure graph are used to match the query frame with the map. To enhance the applicability of the proposed system to real-world robots, we deploy and accelerate the feature detection and matching networks using C++ and NVIDIA TensorRT. Extensive experiments conducted on various datasets demonstrate that our system outperforms other state-of-the-art visual SLAM systems in illumination-challenging environments. Efficiency evaluations show that our system can run at a rate of 73Hz on a PC and 40Hz on an embedded platform. Our implementation is open-sourced: https://github.com/sair-lab/AirSLAM.

Index Terms - Visual SLAM, Mapping, Relocalization.
I. INTRODUCTION

Visual simultaneous localization and mapping (vSLAM) is essential for robot navigation due to its favorable balance between cost and accuracy [1]. Compared to LiDAR SLAM, VSLAM utilizes more cost-effective and compact sensors to achieve accurate localization, thus broadening its range of potential applications [2]. Moreover, cameras can capture richer and more detailed information, which enhances their potential for providing robust localization.

Despite the recent advancements, the present vSLAM systems still struggle with severe lighting conditions [3]-[6], which can be summarized into two categories. First, feature detection and tracking often fail due to drastic changes or low light, severely affecting the quality of the estimated trajectory [7], [8]. Second, when the visual map is reused for relocalization, lighting variations could significantly reduce the success rate [9], [10]. In this paper, we refer to the first issue as the short-term illumination challenge, which impacts pose estimation between two temporally adjacent frames, and the second as the long-term illumination challenge, which affects matching between the query frame and an existing map.

Present methods usually focus on only one of the above challenges. For example, various image enhancement [11]-[13] and image normalization algorithms [14], [15] have been developed to ensure robust tracking. These methods primarily focus on maintaining either global or local brightness consistency, yet they often fall short of handling all types of challenging lighting conditions [16]. Some systems have addressed this issue by training a VO or SLAM network on large datasets containing diverse lighting conditions [17]-[19]. However, they have difficulty producing a map suitable for long-term localization. Some methods can provide illumination-robust relocalization, but they usually require map building under good lighting conditions [20], [21]. In real-world robot applications, these two challenges often arise simultaneously, necessitating a unified system capable of addressing both.

Furthermore, many of the aforementioned systems incorporate intricate neural networks, relying on powerful GPUs to run in real-time. They lack the efficiency necessary for deployment on resource-constrained platforms, such as warehouse robots. These limitations impede the transition of vSLAM from laboratory research to industrial applications.

In response to these gaps, this paper introduces AirSLAM. Observing that line features can improve the accuracy and robustness of vSLAM systems [5], [22], [23], we integrate both point and line features for tracking, mapping, optimization, and relocalization. To achieve a balance between efficiency and performance, we design our system as a hybrid system, employing learning-based methods for feature detection and matching, and traditional geometric approaches for pose and map optimization. Additionally, to enhance the efficiency of feature detection, we developed a unified model capable of simultaneously detecting point and line features. We also address long-term localization challenges by proposing a multi-stage relocalization strategy, which effectively reuses our point-line map. In summary, our contributions include:

    We propose a novel point-line-based vSLAM system that combines the efficiency of traditional optimization techniques with the robustness of learning-based methods. Our system is resilient to both short-term and long-term illumination challenges while remaining efficient enough for deployment on embedded platforms.

    We develop a unified model for both keypoint and line detection, which we call PLNet. To our knowledge, PLNet is the first model capable of simultaneously detecting both point and line features. Furthermore, we associate these two types of features and jointly utilize them for tracking, mapping, and relocalization tasks.

    We propose a multi-stage relocalization method based on both point and line features, utilizing both appearance and geometry information. This method can provide fast and illumination-robust localization in an existing visual map using only a single image.

    We conduct extensive experiments to demonstrate the efficiency and effectiveness of the proposed methods. The results show that our system achieves accurate and robust mapping and relocalization performance under various illumination-challenging conditions. Additionally, our system is also very efficient. It runs at a rate of 73Hz on a PC and 40Hz on an embedded platform.

In addition, our engineering contributions include deploying and accelerating feature detection and matching networks using C++ and NVIDIA TensorRT, facilitating their deployment on real robots. We release all the source code at https://github.com/sair-lab/AirSLAM to benefit the community.

This paper extends our conference paper, AirVO [22]. AirVO utilizes SuperPoint [24] and LSD [25] for feature detection, and SuperGlue [26] for feature matching. It achieves remarkable performance in environments with changing illumination. However, as a visual-only odometry, it primarily addresses short-term illumination challenges and cannot reuse a map for drift-free relocalization. Additionally, despite carefully designed post-processing operations, the modified LSD is still not stable enough for long-term localization. It relies on image gradient information rather than environmental structural information, rendering it susceptible to varying lighting conditions. In this version, we introduce substantial improvements, including:

    We design a unified CNN to detect both point and line features, enhancing the stability of feature detection in illumination-challenging environments. Additionally, the more efficient LightGlue [27] is used for feature matching.

    We extend our system to support both stereo-only and stereo-inertial data, increasing its reliability when an inertial measurement unit (IMU) is available.

    We incorporate loop closure detection and map optimization, forming a complete vSLAM system.

    We design a multi-stage relocalization module based on both point and line features, enabling our system to effectively handle long-term illumination challenges.

The remainder of this article is organized as follows. In Section II, we discuss the relevant literature. Section III-B presents an overview of the complete system pipeline. The proposed PLNet is presented in Section IV. In Section V, we introduce the visual-inertial odometry based on PLNet. Section VI shows how to optimize the map offline and reuse it online. The detailed experimental results are presented in Section VII to verify the efficiency, accuracy, and robustness of AirSLAM. This article is concluded in Section VIII.
II. RELATED WORK
A. Keypoint and Line Detection for vSLAM
1) Keypoint Detection

Various handcrafted keypoint features e.g., ORB [28], FAST [29], and BRISK [30], have been proposed and applied to VO and vSLAM systems. They are usually efficient but not robust enough in challenging environments [9], [26]. With the development of deep learning techniques, more and more learning-based features are proposed and used to replace the handcrafted features in vSLAM systems. Rong et al. [31] introduce TFeat network [32] to extract descriptors for FAST corners and apply it to a traditional vSLAM pipeline. Tang et al. [33] use a neural network to extract robust keypoints and binary feature descriptors with the same shape as the ORB. Han et al. [34] combine SuperPoint [24] feature extractor with a traditional back-end. Bruno et al. proposed LIFT-SLAM [35], where they use LIFT [36] to extract features. Li et al. [37] replace the ORB feature with SuperPoint in ORB-SLAM2 and optimize the feature extraction with the Intel OpenVINO toolkit. Zhan et al. [38] proposed a new self-supervised training scheme for learning-based features, using bundle adjustment and bi-level optimization as a supervision signal. Some other learning-based features, e.g., R2D2 [39] and DISK [40], and D2Net [41], are also being attempted to be applied to vSLAM systems, although they are not yet efficient enough [42], [43].
2) Line Detection

Currently, most point-line-based vSLAM systems use the LSD [25] or EDLines [44] to detect line features because of their good efficiency [5], [45]-[48]. Although many learning-based line detection methods, e.g., SOLD2 [49], AirLine [50], and HAWP [51], have been proposed and shown better robustness in challenging environments, they are difficult to apply to real-time vSLAM systems due to lacking efficiency. For example, Kannapiran et al. propose StereoVO [23], where they choose SuperPoint [24] and SOLD2 [49] to detect keypoints and line segments, respectively. Despite achieving good performance in dynamic lighting conditions, StereoVO can only run at a rate of about 7Hz on a good GPU.
B. Short-Term Illumination Challenge

Several handcrafted methods have been proposed to improve the robustness of VO and vSLAM to challenging illumination. DSO [52] models brightness changes and jointly optimizes camera poses and photometric parameters. DRMS [11] and AFE-ORB-SLAM [12] utilize various image enhancements. Some systems try different methods, such as ZNCC, the locally-scaled sum of squared differences (LSSD), and dense descriptor computation, to achieve robust tracking [14], [15], [53]. These methods mainly focus on either global or local illumination change for all kinds of images, however, lighting conditions often affect the scene differently in different areas [16]. Other related methods include that of Huang and Liu [54], which presents a multi-feature extraction algorithm to extract two kinds of image features when a single-feature algorithm fails to extract enough feature points. Kim et al. [55] employ a patch-based affine illumination model during direct motion estimation. Chen et al. [56] minimize the normalized information distance with nonlinear least square optimization for image registration. Alismail et al. [57] propose a binary feature descriptor using a descriptor assumption to avoid brightness constancy.

Compared with handcrafted methods, learning-based methods have shown better performance. Savinykh et al. [8] propose DarkSLAM, where Generative Adversarial Network (GAN) [58] is used to enhance input images. Pratap Singh et al. [59] compare different learning-based image enhancement methods for vSLAM in low-light environments. TartanVO [17], DROID-SLAM [18], and iSLAM [19] train their VO or SLAM networks on the TartanAir dataset [60], which is a large simulation dataset that contains various lighting conditions, therefore, they are very robust in challenging environments. However, they usually require good GPUs and long training times. Besides, DROID-SLAM runs very slowly and is difficult to apply to real-time applications on resource-constrained platforms. Tartan VO and ISLAM are more efficient, but they cannot achieve performance as accurately as traditional vSLAM systems.
C. Long-Term Illumination Challenge

Currently, most SLAM systems still use the bag of words (BoW) [61] for loop closure detection and relocalization due to its good balance between efficiency and effectiveness [4], [62], [63]. To make the relocalization more robust to large illumination variations, Labbé et al. [20] propose the multi-session relocalization method, where they combine multiple maps generated at different times and in various illumination conditions. DXSLAM [37] uses NetVLAD [64] for the coarse image retrieval and SuperPoint with a binary descriptor for keypoint matching between the query frame and candidates.

Another similar task in the robotics and computer vision communities is the visual place recognition (VPR) problem, where many researchers handle the localization problem with image retrieval methods [64], [65]. These VPR solutions try to find images most similar to the query image from a database. They usually cannot directly provide accurate pose estimation which is needed in robot applications. Sarlin et al. address this and propose Hloc [9]. They use a global retrieval to obtain several candidates and match local features within those candidates. The Hloc toolbox has integrated many image retrieval methods, local feature extractors, and matching methods, and it is currently the SOTA system. Yan et al. [66] propose a long-term visual localization method for mobile platforms, however, they rely on other sensors, e.g., GPS, compass, and gravity sensor, for the coarse location retrieval.
III. SYSTEM OVERVIEW
A. Notations

In this paper, R represents the set of real numbers, and Rm denotes the m-dimensional real vector space. The transpose of a vector or matrix is written as (⋅)⊤. For a vector x∈Rm, ∣∣x∣∣ represents its Euclidean norm, while ∣∣x∣∣Σ2​ is shorthand for x⊤Σx. The transformation, rotation, and translation from the b-coordinate system to the a-coordinate system are denoted by Tab​∈SE(3), Rab​∈SO(3), and tab​∈R3, respectively. We use (⋅)c​ to indicate a vector in the camera frame and (⋅)w​ to indicate a vector in the global world frame. Throughout the paper, super/subscripts may be omitted for brevity, as long as the meaning remains unambiguous within the given context.
B. System Architecture

We believe that a practical vSLAM system should possess the following features:

    High efficiency. The system should have real-time performance on resource-constrained platforms.

    Scalability. The system should be easily extensible for various purposes and real-world applications.

    Easy to deploy. The system should be easy to deploy on real robots and capable of achieving robust localization.

Therefore, we design a system as shown in Fig. 1. The proposed system is a hybrid system as we need the robustness of data-driven approaches and the accuracy of geometric methods. It consists of three main components: stereo VO/VIO, offline map optimization, and lightweight relocalization.

    Image Description (Fig. 1): This figure is a flowchart illustrating the proposed system's three main parts: Online, Offline, and Map.

        Inputs: Two images of a foggy outdoor scene are fed into the system.

        Online Module: This module contains two sub-processes. The first, "Mapping," takes the input images and uses a "Stereo VO/VIO" component to produce an "Initial Map". The second, "Map Reuse," takes an "Optimized Map" and performs "Lightweight Relocalization".

        Offline Module: The "Initial Map" from the online module is processed offline. This involves "Loop Detection," "Map Merging," and "Global Bundle Adjustment." The output is the "Optimized Map," which is then used by the "Map Reuse" process in the online module.
        This architecture separates the real-time processing (VO/VIO and relocalization) from the computationally intensive map optimization, which can be done offline.

(1) Stereo VO/VIO: We propose a point-line-based visual odometry that can handle both stereo and stereo-inertial inputs. (2) Offline map optimization: We implement several commonly used plugins, such as loop detection, pose graph optimization, and global bundle adjustment. The system is easily extensible for other map-processing purposes by adding customized plugins. For example, we have implemented a plugin to train a scene-dependent junction vocabulary using the endpoints of line features, which is utilized in our lightweight multi-stage relocalization. (3) Lightweight relocalization: We propose a multi-stage relocalization method that improves efficiency while maintaining effectiveness. In the first stage, keypoints and line features are detected using the proposed PLNet, and several candidates are retrieved using a keypoint vocabulary trained on a large dataset. In the second stage, most false candidates are quickly filtered out using a scene-dependent

IV. FEATURE DETECTION
---

### A. Motivation

With advancements in deep learning technology, learning-based feature detection methods have demonstrated more stable performance in illumination-challenging environments compared to traditional methods. However, existing point-line-based VO/VIO and SLAM systems typically detect keypoints and line features separately. While it is acceptable for handcrafted methods due to their efficiency, the simultaneous application of keypoint detection and line detection networks in VO/VIO or SLAM systems, especially in stereo configurations, often hinders real-time performance on resource-constrained platforms. Consequently, we aim to design an efficient unified model that can detect keypoints and line features concurrently.

However, achieving a unified model for keypoint and line detection is challenging, as these tasks typically require different real-image datasets and training procedures. Keypoint detection models are generally trained on large datasets comprising diverse images and depend on either a boosting step or the correspondences of image pairs for training. For line detection, we find wireframe parsing methods can provide stronger geometric cues than the self-supervised models as they are able to detect longer and more complete lines. The wireframe includes all prominent straight lines and their junctions within the scene, providing an efficient and accurate representation of large-scale geometry and object shapes. However, these methods are trained on the Wireframe dataset, which is limited in size with only 5,462 discontinuous images. In the following sections, we will address this challenge and demonstrate how to train a unified model capable of performing both tasks. It is important to note that in this paper, the term "line detection" refers specifically to the wireframe parsing task.

### B. Architecture Design

*Description of Figure 2: The image displays four panels. The top-left shows an "Original Image" of a living room. The top-right shows a "Feature Map" of the same image in grayscale, highlighting edges and corners. The bottom-left, labeled "Keypoint Detection," shows the original image with numerous red keypoints detected on objects and structural elements. The bottom-right, labeled "Line Detection," shows the same image with green structural lines detected along edges of walls, windows, and furniture.*

As shown in Fig. 2, we have two findings when visualizing the results of the keypoint and line detection networks: (1) Most junctions (endpoints of lines) detected by the line detection model are also selected as keypoints by the keypoint detection model. (2) The feature maps outputted by the keypoint detection model contain the edge information. Therefore, we argue that a line detection model can be built on the backbone of a pre-trained keypoint detection model. Based on this assumption, we design the PLNet to detect keypoints and lines in a unified framework.

*Description of Figure 3: The image shows the framework of the proposed PLNet. An "Input" image goes into a shared "Backbone" network. The output from the backbone is then fed into two separate modules: a "Line Module" which produces line detections, and a "Keypoint Module" which produces keypoint detections.*

As shown in Fig. 3, it consists of the shared backbone, the keypoint module, and the line module.

**Backbone:** We follow SuperPoint to design the backbone for its good efficiency and effectiveness. It uses 8 convolutional layers and 3 max-pooling layers. The input is the grayscale image sized H×W. The outputs are H×W×64, H/2×W/2×64, H/4×W/4×128, H/8×W/8×128 feature maps.

**Keypoint Module:** We also follow SuperPoint to design the keypoint detection header. It has two branches: the score branch and the descriptor branch. The inputs are H/8×W/8×128 feature maps outputted by the backbone. The score branch outputs a tensor sized H/8×W/8×65. The 65 channels correspond to an 8×8 grid region and a dustbin indicating no keypoint. The tensor is processed by a softmax and then resized to H×W. The descriptor branch outputs a tensor sized H/8×W/8×256, which is used for interpolation to compute descriptors of keypoints.

**Line Module:** This module takes feature maps from the backbone as inputs. It consists of a U-Net-like CNN and the line detection header. We modify the U-Net to make it contain fewer convolutional layers and thus be more efficient. The U-Net-like CNN is to increase the receptive field as detecting lines requires a larger receptive field than detecting keypoints. The EPD LOIAlign is used to process the outputs of the line module and finally outputs junctions and lines.

### C. Network Training

Due to the training problem described in Section IV-A and the assumption in Section IV-B, we train our PLNet in two rounds. In the first round, only the backbone and the keypoint detection module are trained, which means we need to train a keypoint detection network. In the second round, the backbone and the keypoint detection module are fixed, and we only train the line detection module on the Wireframe dataset. We skip the details of the first round as they are very similar to [24]. Instead, we present the training of the line detection module.

*Description of Figure 4: The image illustrates how a line segment is encoded. On the left, a line segment on a countertop is highlighted. On the right, a diagram shows the parameters used for encoding: a line segment defined by endpoints x1 and x2, and a point p within its attraction region. The parameters are d (distance from p to the line), θ (angle of the line), and angles θ1 and θ2.*

**Line Encoding:** We adopt the attraction region field to encode line segments. As shown in Fig. 4, for a line segment $l=(x_1, x_2)$ where $x_1$ and $x_2$ are two endpoints of l, and a point p in the attraction region of l, four parameters and p are used to encode l:
$p(l)=(d, \theta, \theta_1, \theta_2)$, (1)
where d is the distance from p to l, $\theta$ is the angle between l and the v-axis of the image, $\theta_1$ is the angle between $px_1$ and the perpendicular line from p to l, and $\theta_2$ is the angle between $px_2$ and the perpendicular line. The network can predict these four parameters for point p and then l can be decoded through:
$l=d \cdot \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} 1 & 1 \\ \tan\theta_1 & \tan\theta_2 \end{bmatrix} + \begin{bmatrix} p & p \end{bmatrix}$ (2)

**Line Prediction:** The line detection module outputs a tensor sized H/4×W/4×4 to predict parameters in (1) and a heatmap to predict junctions. For each decoded line segment by (2), two junctions closest to its endpoints will be selected to form a line proposal with it. Proposals with the same junctions will be deduplicated and only one is retained. Then the EPD LOIAlign and a head classifier are applied to decide whether the line proposal is a true line feature.

**Line Module Training:** We use the L1 loss to supervise the prediction of parameters in (1) and the binary cross-entropy loss to supervise the junction heatmap and the head classifier. The total loss is the sum of them.

*Description of Figure 5: This image shows an "Original Image" and seven augmented versions used for training PLNet. The augmentations are: "Random Contrast," "Random Brightness," "Motion Blur," "Fog," "Shade," "Speckle Noise," and "Gaussian Noise."*

As shown in Fig. 5, to improve the robustness of line detection in illumination-challenging environments, seven types of photometric data augmentation are applied to process training images. The training uses the ADAM optimizer with the learning rate $lr=4e-4$ in the first 35 epochs and $lr=4e-5$ in the last 5 epochs.

---
V. STEREO VISUAL ODOMETRY
---

### A. Overview

*Description of Figure 6: This is a flowchart of the visual-inertial odometry system. Inputs are "Left Image," "Right Image," and optional "IMU" data. The top part (front-end thread) shows feature detection, point-line association, and matching. The bottom part (back-end thread) shows IMU preintegration, initial pose estimation, triangulation, local bundle adjustment, and map updating, ultimately producing a "Frame" output.*

The proposed point-line-based stereo visual odometry is shown in Fig. 6. It is a hybrid VO system utilizing both the learning-based front-end and the traditional optimization backend. For each stereo image pair, we first employ the proposed PLNet to extract keypoints and line features. Then a GNN (LightGlue) is used to match keypoints. In parallel, we associate line features with keypoints and match them using the keypoint matching results. After that, we perform an initial pose estimation and reject outliers. Based on the results, we triangulate the 2D features of keyframes and insert them into the map. Finally, the local bundle adjustment will be performed to optimize points, lines, and keyframe poses. In the meantime, if an IMU is accessible, its measurements will be processed using the IMU preintegration method, and added to the initial pose estimation and local bundle adjustment.

Applying both learning-based feature detection and matching methods to the stereo VO is time-consuming. Therefore, to improve efficiency, the following three techniques are utilized in our system. (1) For keyframes, we extract features on both left and right images and perform stereo matching to estimate the real scale. But for non-keyframes, we only process the left image. Besides, we use some lenient criteria to make the selected keyframes in our system very sparse, so the runtime and resource consumption of feature detection and matching in our system are close to that of a monocular system. (2) We convert the inference code of the CNN and GNN from Python to C++ and deploy them using ONNX and NVIDIA TensorRT, where the 16-bit floating-point arithmetic replaces the 32-bit floating-point arithmetic. (3) We design a multi-thread pipeline. A producer-consumer model is used to split the system into two main threads, i.e., the front-end thread and the backend thread. The front-end thread extracts and matches features while the backend thread performs the initial pose estimation, keyframe insertion, and local bundle adjustment.

### B. Feature Matching

We use LightGlue to match keypoints. For line features, most of the current VO and SLAM systems use the LBD algorithm or tracking sample points to match them. However, the LBD algorithm extracts the descriptor from a local band region of the line, so it suffers from unstable line detection due to challenging illumination or viewpoint changes. Tracking sample points can match the line detected with different lengths in two frames, but current SLAM systems usually use optical flow to track the sample points, which have a bad performance when the light conditions change rapidly or violently. Some learning-based line feature descriptors are also proposed, however, they are rarely used in current SLAM systems due to the increased time complexity.

Therefore, to address both the effectiveness problem and efficiency problem, we design a fast and robust line-matching method for illumination-challenging conditions. First, we associate keypoints with line segments through their distances. Assume that M keypoints and N line segments are detected on the image, where the i-th keypoint is denoted as $p_i = (x_i, y_i)$ and the j-th line segment is denoted as $l_j = (A_j, B_j, C_j, x_{j,1}, y_{j,1}, x_{j,2}, y_{j,2})$, where $(A_j, B_j, C_j)$ are line parameters of $l_j$ and $(x_{j,1}, y_{j,1}, x_{j,2}, y_{j,2})$ are the endpoints. We first compute the distance between $p_i$ and $l_j$ through:
$d_{ij} = d(p_i, l_j) = \frac{|A_j \cdot x_i + B_j \cdot y_i + C_j|}{\sqrt{A_j^2 + B_j^2}}$ (3)

If $d_{ij} < 3$ and the projection of $p_i$ on the coordinate axis lies within the projections of line segment endpoints, i.e., $\min(x_{j,1}, x_{j,2}) \le x_i \le \max(x_{j,1}, x_{j,2})$ or $\min(y_{j,1}, y_{j,2}) \le y_i \le \max(y_{j,1}, y_{j,2})$, we will say $p_i$ belongs to $l_j$. Then the line segments on two images can be matched based on the point-matching result of these two images. For $l_{k,m}$ on image k and $l_{k+1,n}$ on image k+1, we compute a score $S_{mn}$ to represent the confidence of that they are the same line:
$S_{mn} = \frac{N_{pm}}{\min(N_{k,m}, N_{k+1,n})}$ (4)
where $N_{pm}$ is the matching number between point features belonging to $l_{k,m}$ and point features belonging to $l_{k+1,n}$. $N_{k,m}$ and $N_{k+1,n}$ are the numbers of point features belonging to $l_{k,m}$ and $l_{k+1,n}$, respectively. Then if $S_{mn} > \delta_S$ and $N_{pm} > \delta_N$ where $\delta_S$ and $\delta_N$ are two preset thresholds, we will regard $l_{k,m}$ and $l_{k+1,n}$ as the same line. This coupled feature matching method allows our line matching to share the robust performance of keypoint matching while being highly efficient due to that it does not need another line-matching network.

### C. 3D Feature Processing

In this part, we will introduce our 3D feature processing methods, including 3D feature representation, triangulation, i.e., constructing 3D features from 2D features, and re-projection, i.e., projecting 3D features to the image plane.

For 3D point processing, a 3D point is denoted as $X \in \mathbb{R}^3$. LightGlue is utilized to match keypoints between the left and right images. Successfully matched keypoints are triangulated using stereo disparity information, while unmatched keypoints are triangulated leveraging multi-frame observations. The projection of 3D points onto the image plane is modeled using either the pinhole camera model or the fisheye camera model, depending on the specific camera in use. We skip the details of 3D point processing in our system as they are similar to other point-based VO and SLAM systems.

On the contrary, compared with 3D points, 3D lines have more degrees of freedom, and they are easier to degenerate when being triangulated. Therefore, the 3D line processing will be illustrated in detail.

**1) 3D Line Representation:** We use Plücker coordinates to represent a 3D spatial line:
$L = \begin{bmatrix} n \\ v \end{bmatrix} \in \mathbb{R}^6$ (5)
where v is the direction vector of the line and n is the normal vector of the plane determined by the line and the origin. Plücker coordinates are used for 3D line triangulation, transformation, and projection. It is over-parameterized because it is a 6-dimensional vector, but a 3D line has only four degrees of freedom. In the graph optimization stage, the extra degrees of freedom will increase the computational cost and cause the numerical instability of the system. Therefore, we also use orthonormal representation to represent a 3D line:
$(U, W) \in SO(3) \times SO(2)$ (6)
The relationship between Plücker coordinates and orthonormal representation is similar to $SO(3)$ and $so(3)$. Orthonormal representation can be obtained from Plücker coordinates by:
$L = [\dots]$ (7) *[Note: Formula incomplete in source]*
where $\Sigma_{3x2}$ is a diagonal matrix and its two non-zero entries defined up to scale can be represented by an $SO(2)$ matrix:
$W = \frac{1}{\sqrt{||n||^2 + ||v||^2}} \begin{bmatrix} ||n|| & -||v|| \\ ||v|| & ||n|| \end{bmatrix} \in SO(2)$ (8)
In practice, this conversion can be done simply and quickly with the QR decomposition.

**2) Triangulation:** Triangulation is to initialize a 3D line from two or more 2D line features. In our system, we use two methods to triangulate a 3D line. The first is similar to the line triangulation algorithm B in [77], where the pose of a 3D line can be computed from two planes. To achieve this, we select two line segments, $l_1$ and $l_2$, on two images, which are two observations of a 3D line. Note that the two images can come from the stereo pair of the same keyframe or two different keyframes. $l_1$ and $l_2$ can be back-projected and construct two 3D planes, $\pi_1$ and $\pi_2$. Then the 3D line can be regarded as the intersection of $\pi_1$ and $\pi_2$.

However, triangulating a 3D line is more difficult than triangulating a 3D point, because it suffers more from degenerate motions. Therefore, we also employ a second line triangulation method if the above method fails, where points are utilized to compute the 3D line. In Section V-B, we have associated point features with line features. So to initialize a 3D line, two triangulated points $X_1$ and $X_2$, which belong to this line and have the shortest distance from this line on the image plane are selected. Then the Plücker coordinates of this line can be obtained through:
$L = \begin{bmatrix} n \\ v \end{bmatrix} = \begin{bmatrix} X_1 \times X_2 \\ \frac{X_1 - X_2}{||X_1 - X_2||} \end{bmatrix}$ (9)
This method requires little extra computation because the selected 3D points have been triangulated in the point triangulating stage. It is very efficient and robust.

**3) Re-projection:** Re-projection is used to compute the re-projection errors. We use Plücker coordinates to transform and re-project 3D lines. First, we convert the 3D line from the world frame to the camera frame:
$L_c = \begin{bmatrix} n_c \\ v_c \end{bmatrix} = H_{cw} L_w = \begin{bmatrix} R_{cw} & [t_{cw}]_x R_{cw} \\ 0 & R_{cw} \end{bmatrix} \begin{bmatrix} n_w \\ v_w \end{bmatrix}$ (10)
where $L_c$ and $L_w$ are Plücker coordinates of 3D line in the camera frame and world frame, respectively. $R_{cw} \in SO(3)$ is the rotation matrix from world frame to camera frame and $t_{cw} \in \mathbb{R}^3$ is the translation vector. $[\cdot]_x$ denotes the skew-symmetric matrix of a vector and $H_{cw}$ is the transformation matrix of 3D lines from world frame to camera frame.
Then the 3D line $L_c$ can be projected to the image plane through a line projection matrix $P_c$:
$l = \begin{bmatrix} A \\ B \\ C \end{bmatrix} = P_c L_{c[:3]} = \begin{bmatrix} f_x & 0 & 0 \\ 0 & f_y & 0 \\ -f_y c_x & -f_x c_y & f_x f_y \end{bmatrix} n_c$ (11)
where $l = [A\; B\; C]^T$ is the re-projected 2D line on image plane. $L_{c[:3]}$ donates the first three rows of vector $L_c$.

### D. Keyframe Selection

Observing that the learning-based data association method used in our system is able to track two frames that have a large baseline, so different from the frame-by-frame tracking strategy used in other VO or SLAM systems, we only match the current frame with the last keyframe. We argue this strategy can reduce the accumulated tracking error.

Therefore, the keyframe selection is essential for our system. On the one hand, as described in Section V-A, we want to make keyframes sparse to reduce the consumption of computational resources. On the other hand, the sparser the keyframes, the more likely tracking failure happens. To balance the efficiency and the tracking robustness, a frame will be selected as a keyframe if any of the following conditions is satisfied:
* The tracked features are less than $\alpha_1 \cdot N_s$.
* The average parallax of tracked features between the current frame and the last keyframe is larger than $\alpha_2 \cdot \sqrt{WH}$.
* The number of tracked features is less than $N_{kf}$.

In the above, $\alpha_1$, $\alpha_2$, and $N_{kf}$ are all preset thresholds. $N_s$ is the number of detected features. W and H respectively represent the width and height of the input image.

### E. Local Graph Optimization

To improve the accuracy, we perform the local bundle adjustment when a new keyframe is inserted. $N_o$ latest neighboring keyframes are selected to construct a local graph, where map points, 3D lines, and keyframes are vertices and pose constraints are edges. We use point constraints and line constraints as well as IMU constraints if an IMU is accessible. Their related error terms are defined as follows.

**1) Point Re-projection Error:** If the frame i can observe the 3D map point $X_p$ then the re-projection error is defined as:
$r_{i,X_p} = \tilde{x}_{i,p} - \pi(R_{cw}X_p + t_{cw})$ (12)
where $\tilde{x}_{i,p}$ is the observation of $X_p$ on frame i and $\pi(\cdot)$ represents the camera projection.

**2) Line Re-projection Error:** If the frame i can observe the 3D line $L_q$, then the re-projection error is defined as:
$r_{i,L_q} = e_l(\tilde{l}_{i,q}, P_c(H_{cw}L_q)_{[:3]}) \in \mathbb{R}^2$ (13a)
$e_l(\tilde{l}_{i,q}, l_{i,q}) = [d(\tilde{p}_{i,q1}, l_{i,q})\; d(\tilde{p}_{i,q2}, l_{i,q})]^T$ (13b)
where $\tilde{l}_{i,q}$ is the observation of $L_q$ on frame i, $\tilde{p}_{i,q1}$ and $\tilde{p}_{i,q2}$ are the endpoints of $\tilde{l}_{i,q}$, and $d(p, l)$ is the distance between point p and line l which is computed through (3).

**3) IMU Residuals:** We first follow [72] to pre-integrate IMU measurements between the frame i and the frame j:
$\Delta \tilde{R}_{ij} = \prod_{k=i}^{j-1} \text{Exp}((\tilde{\omega}_k - b_k^g - \eta_k^{gd})\Delta t)$ (14a)
$\Delta \tilde{v}_{ij} = \sum_{k=i}^{j-1} \Delta \tilde{R}_{ik} (\tilde{a}_k - b_k^a - \eta_k^{ad})\Delta t$ (14b)
$\Delta \tilde{p}_{ij} = \sum_{k=i}^{j-1} (\Delta \tilde{v}_{ik} \Delta t + \frac{1}{2} \Delta \tilde{R}_{ik} (\tilde{a}_k - b_k^a - \eta_k^{ad}) \Delta t^2)$ (14c)
where $\tilde{\omega}_k$ and $\tilde{a}_k$ are respectively the angular velocity and the acceleration. $b_k^g$ and $b_k^a$ are biases of the sensor and they are modeled as constants between two keyframes through $b_k^g = b_{k+1}^g$ and $b_k^a = b_{k+1}^a$, $\eta_k^{gd}$ and $\eta_k^{ad}$ are Gaussian noises.
Then IMU residuals are defined as:
$r_{\Delta R_{ij}} = \text{Log}((\Delta \tilde{R}_{ij} \text{Exp}(\frac{\partial \Delta R_{ij}}{\partial b^g} \delta b^g))^T R_i^T R_j)$ (15a)
$r_{\Delta v_{ij}} = R_i^T(v_j - v_i - g\Delta t_{ij}) - (\Delta \bar{v}_{ij} + \frac{\partial \Delta v_{ij}}{\partial b^g} \delta b^g + \frac{\partial \Delta v_{ij}}{\partial b^a} \delta b^a)$ (15b)
$r_{\Delta p_{ij}} = R_i^T(p_j - p_i - v_i \Delta t_{ij} - \frac{1}{2} g \Delta t_{ij}^2) - (\Delta \bar{p}_{ij} + \frac{\partial \Delta p_{ij}}{\partial b^g} \delta b^g + \frac{\partial \Delta p_{ij}}{\partial b^a} \delta b^a)$ (15c)
$r_{b_{ij}} = [(b^g_j - b^g_i)^T (b^a_j - b^a_i)^T]^T$ (15d)
where g is the gravity vector in world coordinates. In our system, we combine the initialization process in [4] and [62] to estimate g and initial values of biases.
The factor graph is optimized by the g2o toolbox. The cost function is defined as:
$E = \sum ||r_{i,X_p}||^2_{\Sigma_p} + \sum ||r_{i,L_q}||^2_{\Sigma_L} + \sum ||r_{\Delta R_{ij}}||^2_{\Sigma_{\Delta R}} + \sum ||r_{\Delta v_{ij}}||^2_{\Sigma_{\Delta v}} + \sum ||r_{\Delta p_{ij}}||^2_{\Sigma_{\Delta p}} + \sum ||r_{b_{ij}}||^2_{\Sigma_b}$ (16a)
We use the Levenberg-Marquardt optimizer to minimize the cost function. The point and line outliers are also rejected in the optimization if their corresponding residuals are too large.

### F. Initial Map

As described in Section III-B, our map is optimized offline. Therefore, keyframes, map points, and 3D lines will be saved to the disk for subsequent optimization when the visual odometry is finished. For each keyframe, we save its index, pose, keypoints, keypoint descriptors, line features, and junctions. The correspondences between 2D features and 3D features are also recorded. To make the map faster to save, load, and transfer across different devices, the above information is stored in binary form, which also makes the initial map much smaller than the raw data. For example, on the OIVIO dataset, our initial map size is only about 2% of the raw data size.

---
VI. MAP OPTIMIZATION AND REUSE
---

### A. Offline Map Optimization

This part aims to process an initial map generated by our VO module and outputs the optimized map that can be used for drift-free relocalization. Our offline map optimization module consists of the following several map-processing plugins.

**1) Loop Closure Detection:** Similar to most current vSLAM systems, we use a coarse-to-fine pipeline to detect loop closures. Our loop closure detection relies on DBoW2 to retrieve candidates and LightGlue to match features. We train a vocabulary for the keypoint detected by our PLNet on a database that contains 35k images. These images are selected from several large datasets that include both indoor and outdoor scenes. The vocabulary has 4 layers, with 10 nodes at each layer, so it contains 10,000 words.

**Coarse Candidate Selection:** This step aims to find three candidates most similar to a keyframe $K_i$ from a set $S_1 = \{K_j | j < i\}$. Note that we do not add keyframes with an index greater than $K_i$ to the set because this may miss some loop pairs. We build a co-visibility graph for all keyframes where two are connected if they observe at least one feature. All keyframes connected with $K_i$ will be first removed from $S_1$. Then we compute a similarity score between $K_i$ and each keyframe in $S_1$ using DBoW2. Only keyframes with a score greater than $0.3 \cdot S_{max}$ will be kept in $S_1$ where $S_{max}$ is the maximum computed score. After that, we group the remaining keyframes. If two keyframes can observe more than 10 features in common, they will be in the same group. For each group, we sum up the scores of the keyframes in this group and use it as the group score. Only the top 3 groups with the highest scores will be retained. Then we select one keyframe with the highest score within the group as the candidate from each group. These three candidates will be processed in the subsequent steps.

**Fine Feature Matching:** For each selected candidate, we match its features with $K_i$. Then the relative pose estimation with outlier rejection will be performed. The candidate will form a valid loop pair with $K_i$ if the inliers exceed 50.

**2) Map Merging:** A 3D feature observed by both frames of a loop pair is usually mistakenly used as two features. Therefore, in this part, we aim to merge the duplicated point and line features observed by loop pairs. For keypoint features, we use the above feature-matching results between loop pairs. If two matched keypoints are associated with two different map points, they will be regarded as duplicated features and only one map point will be retained. The correspondence between 2D keypoints and 3D map points, as well as the connections in the co-visibility graph, will also be updated. For line features, we first associate 3D lines and map points through the 2D-3D feature correspondence and 2D point-line association built in Section V-B. Then we detect 3D line pairs that associate with the same map points. If two 3D lines share more than 3 associated map points, they will be regarded as duplicated and only one 3D line will be retained.

**3) Global Bundle Adjustment:** We perform the global bundle adjustment (GBA) after merging duplicated features. The form of the residuals and loss functions is similar to that in Section V-E; however, unlike Section V-E, all keyframes and features are jointly optimized, and loop closure residuals are also incorporated into the optimization process. In the initial stage of optimization, the re-projection errors of merged features are relatively large due to the VO drift error, so we first iterate 50 times without outlier rejection to optimize the variables to a good rough position, and then iterate another 40 times with outlier rejection. We find that when the map is large, the initial 50 iterations can not optimize the variables to a satisfactory position. To address this, we first perform pose graph optimization (PGO) before the global bundle adjustment if a map contains more than 80k map points. Only the keyframe poses will be adjusted in the PGO and the cost function is defined as follows:
$E_{pgo} = \sum ||\text{Log}(\Delta \bar{T}_{ij}^{-1} T_i^{-1} T_j)||^2_{\Sigma_{ij}}$ (17)
where $T_i \in SE(3)$ and $T_j \in SE(3)$ are poses of $K_i$ and $K_j$, respectively. $\text{Log}(\cdot) = \log(\cdot)^\vee : SE(3) \rightarrow se(3)$ is the Logarithm map proposed in [83]. $K_i$ and $K_j$ should either be adjacent or form a loop pair. After the pose graph optimization, the positions of map points and 3D lines will also be adjusted along with the keyframes in which they are first observed. The systems with online loop detection usually perform the GBA after detecting a new loop, so they undergo repeated GBAs when a scene contains many loops. In contrast, our offline map optimization module only does the GBA after all loop closures are detected, allowing us to reduce the optimization iterations significantly compared with them.

**4) Scene-Dependent Vocabulary:** We train a junction vocabulary aiming to be used for relocalization. The vocabulary is built on the junctions of keyframes in the map so it is scene-dependent. Compared with the keypoint vocabulary trained in Section VI-A1, the database used to train the junction vocabulary is generally much smaller, so we set the number of layers to 3, with 10 nodes in each layer. The junction vocabulary is tiny, i.e., about 1 megabyte, as it only contains 1000 words. Its detailed usage will be introduced in Section VI-B.

**5) Optimized Map:** we save the optimized map for subsequent map reuse. Compared with the initial map in Section V-F, more information is saved such as the bag of words for each keyframe, the global co-visibility graph, and the scene-dependent junction vocabulary. In the meantime, the number of 3D features has decreased due to the fusion of duplicate map points and 3D lines. Therefore, the optimized map occupies a similar memory to the initial map.

### B. Map Reuse

In this part, we present our illumination-robust relocalization using an existing optimized map. In most vSLAM systems, recognizing revisited places typically needs two steps: (1) retrieving $N_{kc}$ keyframe candidates and (2) performing feature matching and estimating relative pose. The second step is usually time-consuming, so selecting a proper $N_{kc}$ is very important. A larger $N_{kc}$ will reduce the system's efficiency while a smaller $N_{kc}$ may prevent the correct candidate from being recalled. For example, in the loop closing module of ORB-SLAM3, only the three most similar keyframes retrieved by DBoW2 are used for better efficiency. It works well as two frames in a loop pair usually have a short time interval and thus the lighting conditions are relatively similar. But for challenging tasks, such as the day/night relocalization problem, retrieving so few candidates usually results in a low recall rate. However, retrieving more candidates needs to perform feature matching and pose estimation more times for each query frame, which makes it difficult to deploy for real-time applications.

To address this problem, we propose an efficient multi-stage relocalization method to make the optimized map usable in different lighting conditions. Our insight is that if most of the false candidates can be quickly filtered out, then the efficiency can be improved while maintaining or even improving the relocalization recall rate. Therefore, we add another step to the two-step pipeline mentioned above. We next introduce the proposed multi-stage pipeline in detail.

**1) The First Step:** This step is to retrieve the similar keyframes in the map that are similar to the query frame. For each input monocular image, we detect keypoints, junctions, and line features using our PLNet. Then a pipeline similar to the "coarse candidate selection" in Section VI-A1 will be executed, but with two differences. The first difference is that we do not filter out candidates using the co-visibility graph as the query frame is not in the graph. The second is that all candidates, not just three, will be retained for the next step.

**2) The Second Step:** This step filters out most of the candidates selected in the first step using junctions and line features. For query frame $K_q$ and each candidate $K_b$, we first match their junctions by finding the same words through the junction vocabulary trained in Section VI-A4. We use $\{(q_i, b_i) | q_i \in K_q, b_i \in K_b\}$ to denote the matching pairs. Then we construct two structure graphs, i.e., $G_q^J$ and $G_b^J$ for $K_q$ and $K_b$, respectively. The vertices are matched junctions, i.e., $V_q^J = \{q_i | q_i \in K_q\}$ and $V_b^J = \{b_i | b_i \in K_b\}$. The related adjacent matrices that describe the connection between vertices are defined as:
$A_q^J = \begin{bmatrix} q_{11} & \cdots & q_{1n} \\ \vdots & \ddots & \vdots \\ q_{n1} & \cdots & q_{nn} \end{bmatrix}, A_b^J = \begin{bmatrix} b_{11} & \cdots & b_{1n} \\ \vdots & \ddots & \vdots \\ b_{n1} & \cdots & b_{nn} \end{bmatrix}$ (18)
where n is the number of junction-matching pairs. $q_{ij}$ is set to 1 if the junction $q_i$ and $q_j$ are two endpoints of the same line, otherwise, it is set to 0. The same goes for $b_{ij}$. Then the graph similarity of $G_q^J$ and $G_b^J$ can be computed through:
$S_{qb}^G = \sum 1 - |q_{ij} - b_{ij}|$ (19)
We also compute a junction similarity score $S_{qb}^J$ using the junction vocabulary and the DBoW2 algorithm. Finally, the similarity score of $K_q$ and $K_b$ is given by combining the keypoint similarity, junction similarity, and structure graph similarity:
$S_{qb} = S_{qb}^K + S_{qb}^J \cdot (1 + \frac{S_{qb}^G}{n})$ (20)
where $S_{qb}^K$ is the keypoint similarity of $K_q$ and $K_b$ computed in the first step. We compute the similarity score with the query frame for each candidate, and only the top 3 candidates with the highest similarity scores will be retained for the next step.

**Analysis:** We next analyze the second step. In the normal two-step pipeline that uses the DBoW method, only appearance information is used to retrieve candidates. The structural information, i.e., the necessity of the consistent spatial distribution of features between the query frame and candidate, is ignored in the first step and only used in the second step. However, in the illumination-challenging scenes, the structural information is essential as it is invariant to lighting conditions. In our second step, a portion of the structural information is utilized to select candidates. First, our PLNet uses the wireframe-parsing method to detect structural lines, which are more stable in illumination-challenging environments. Second, the similarity computed in (20) utilizes both the appearance information and the structural information. Therefore, our system can achieve good performance in illumination-challenging environments although using the efficient DBoW method.

The second step is also highly efficient. On the one hand, junctions are usually much less than keypoints. In normal scenes, our PLNet can detect more than 400 good keypoints but only about 50 junctions. On the other hand, the junction vocabulary is tiny and only contains 1,000 words. Therefore, matching junctions using DBoW2, constructing junction graphs, and computing similarity scores are all executed very efficiently. The experiment shows that the second step can be done within 0.7ms. More results will be presented in Section VII.

**3) The Third Step:** The third step aims to estimate the pose of the query frame. We first use LightGlue to match features between the query frame and the retained candidates. The candidate with the most matching inliers will be selected as the best candidate. Then based on the matching results of the query frame and the best candidate, we can associate the query keypoints with map points. FFinally, a PnP problem is solved with RANSAC to estimate the pose. The pose will be
considered valid if the inliers exceed 20.