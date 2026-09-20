<div align="center">

# Shivpratap Singh Panwar

### I teach humanoid robots to move and machines to see — from scratch, on hardware that actually ships.

**Computer Vision Research Engineer** · Humanoid RL · Robot Fleet Perception · Edge Inference · Medical AI

[![Email](https://img.shields.io/badge/Email-shivpratapsinghpanwar19%40gmail.com-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:shivpratapsinghpanwar19@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Shivpratap_Singh_Panwar-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/shivpratap-singh-panwar/)
[![Kaggle](https://img.shields.io/badge/Kaggle-50%2B_notebooks-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/shivpratap0007)
[![Portfolio](https://img.shields.io/badge/Portfolio-shivpratapsinghpanwar.github.io-00918A?style=flat-square&logo=githubpages&logoColor=white)](https://shivpratapsinghpanwar.github.io)

<br>

| 🏭 Multi-model serving | ⚡ INT4 on Jetson | 🎥 Multi-cam robot fleets | 🤖 23-DoF humanoid | 📦 Shipped on PyPI |
|:---:|:---:|:---:|:---:|:---:|
| **in production**, on one GPU | full quantization stack | perception at fleet scale | locomotion **from scratch** | first-author IEEE · Springer |

</div>

---

## 🏭 Production vision systems

The part of computer vision that is not the model: getting many models to run together, on real
hardware, against numbers that hold up. Patterns I build and own:

```mermaid
graph LR
    A["Camera ingest<br/>cable · RTSP · file"] --> B["Triton model repository<br/>poll-mode, hot-pluggable"]
    B --> C["Several detection systems<br/>sharing one pipeline"]
    C --> D["Operator dashboard<br/>FastAPI + React, live alerts"]
    D -->|"staged weights go live without a redeploy"| B
```

- **Multi-model serving.** Several independent detection systems share one pipeline and one GPU
  budget. A system with no trained model yet reports itself as *awaiting model* and comes online the
  moment weights are staged — no redeploy, no downtime.
- **Immutable dataset versioning.** Every version is a complete self-contained snapshot: physical
  image copies, a canonical manifest, a full audit report, and ready-to-use COCO **and** Pascal-VOC
  exports. Ingesting new data never mutates a previous version, so any result stays reproducible
  against the exact data that produced it.
- **One leaderboard for every run**, ranking models on per-size-band mask recall, false positives and
  median per-frame latency together — so accuracy and speed get compared in a single table instead
  of argued about.
- **Licence-aware model selection.** Permissive-only registries (torch/torchvision, RF-DETR-Seg) when
  a deployment context rules out copyleft frameworks — licence constraints are part of the
  engineering, not an afterthought.

**Zero-copy perception pipelines** on Jetson alongside it: a pre-allocated shared-memory frame ring
where pixels are written once and consumed under read-only leases, a latest-wins policy that keeps
lag bounded and **counts every dropped frame**, and per-service quarantine so one failing model
never takes the pipeline down.

## 🤖 Humanoid locomotion — from scratch, and still going

An ongoing program training **whole-body locomotion for the Unitree G1 (23-DoF)** with **no pretrained policy, no imitation data — reward design up**. Task registry, gait mechanics, perturbation curricula, and the training infrastructure are all custom:

```mermaid
graph LR
    A["Task & reward design<br/>(custom registry)"] --> B["Massively parallel sim<br/>MuJoCo Warp · mjlab"]
    B --> C["PPO · RSL-RL<br/>multi-GPU T4"]
    C --> D["Checkpoint / resume<br/>across session limits"]
    D --> E["Failure diagnosis<br/>dedicated analysis runs"]
    E -->|curriculum update| A
```

- **Running that is actually running** — a speed-adaptive gait clock shortens the cycle with commanded velocity and pushes stance fraction below 0.5, opening a true **flight phase**. Stock fixed clocks make "running" an unlearnable fast walk; mine makes it a gait.
- **Take hits from any direction** — perturbation curriculum mixing instantaneous velocity impulses with **sustained horizontal drags from uniformly random headings**. The policy doesn't memorize one shove; it learns balance recovery as a skill.
- **Reverse locomotion as a first-class command** (`lin_vel_x ∈ [−1.2, 2.0]`) — walking backwards is trained, not hoped for.
- **Anti-cheat reward shaping** — standing-still and feet-air terms that stop the classic failure mode of a robot marching in place to farm gait rewards.
- **Debugging like an engineer, not a gambler** — when a policy failed backward pushes, it got its own isolation run, diagnosis, and targeted curriculum fix before retraining. Every experiment checkpointed and resumable across free-tier GPU session limits.
- Foundation-model track: [custom design work](https://github.com/shivpratapsinghpanwar/tau-0-vla_Custom_Design) on [τ0-VLA](https://github.com/sii-research/tau-0-vla) — world-model-guided test-time computation for robot control. Base stack: [my fork](https://github.com/shivpratapsinghpanwar/unitree_rl_mjlab_custom_robot) of Unitree's RL suite on [mjlab](https://github.com/mujocolab/mjlab)/MuJoCo Warp.

## 🎥 Perception for robot fleets

At **Kody Technolab** (full-stack robotics company) I build the vision layer for **multi-camera robots operating as fleets** — security, advertisement, data-gathering and assistant platforms:

```mermaid
graph LR
    A["Multi-camera ingest<br/>sync & calibration"] --> B["Multi-task perception<br/>detect · segment · pose · depth"]
    B --> C["Quantize<br/>FP16 → INT4"]
    C --> D["Edge deploy<br/>Jetson · Android robots"]
    D --> E["Fleet in production"]
    E -->|"data flywheel: field footage → retraining"| A
```

- Multi-task heads sharing one latency budget: detection, instance segmentation, pose and depth estimation running together on embedded compute.
- **The whole quantization ladder** — FP16 / INT8 / UINT8 / **UINT4** via TensorRT, ONNX Runtime, TFLite, OpenVINO — chosen per platform, per model, per latency budget.
- Pipelines that have processed **500K+ images and 100+ hours of robot video**; preprocessing time cut **40%**, production throughput up **~30%**.
- Model strategy per constraint: YOLO family, SAM-1/2/3, D-FINE, RF-DETR, MobileOne×ArcFace, custom architectures — the right tool for the hardware in the room, with **classical projective geometry as the fallback where deep learning runs out of data**.

## 🏥 Vision where none exists — medical AI

Building perception for **medical robotics targeting rare anomalies** — conditions with **no existing CV solution and almost no training data** when development starts. That's the point: low-data strategies, geometry-first fallbacks where deep learning runs out of examples, and rigorous evaluation on curated clinical splits.

## 🏭 [Synthetic Data Factory](https://github.com/shivpratapsinghpanwar/Synthetic_Data_Factory)

When real medical data runs out, I manufacture it — and **measure whether it actually helps, not whether it looks pretty**:

- Pluggable generative backends: SD 1.5 + per-class LoRA, and a **from-scratch DDPM with zero natural-image prior** for sensitive domains.
- Every synthetic image is provenance-tracked and screened for **memorization of real patient images** before it may train anything — privacy treated as a hard gate, not a footnote.
- **Paired multi-seed A/B on HAM10000** — 3 seeds, evaluated on 1,508 real images never touched by generation or model selection. Reported as measured: **macro-F1 +0.005 ± 0.019, consistent with zero.** One augmented class cleared its own spread in every seed (vascular lesion F1 **+0.050 ± 0.013**). The largest effect in the experiment was **off-target** — melanoma recall **+0.116 ± 0.048** on a class that was *never augmented* — so the writeup attributes it to class rebalancing rather than synthetic realism, and specifies the balance-matched control that would separate the two. The flattering headline was available; it is not claimed ([full measured results](https://github.com/shivpratapsinghpanwar/Synthetic_Data_Factory/blob/main/docs/results.md)).
- The entire train→evaluate loop executes remotely on free Kaggle GPUs through a **git-pinned execution runner built for autonomous agent iteration** — every run reproducible to the commit.


## 🧪 [data-doctor](https://github.com/shivpratapsinghpanwar/data-doctor) — *diagnose vision datasets before they lie to you*

```bash
pip install vision-data-doctor
```

A delivered production train/val split turned out to contain **81 groups of byte-identical images
spanning both sides**. 40% of the validation set was leaked, the reference model's 0.75 recall did
not survive an honest split, and **nothing in the training stack had warned**. This makes that a
10-second CI check instead of a post-mortem:

- **Hashes nominate; only pixels convict.** SHA-256 catches exact copies; two perceptual hash
  families over all 8 dihedral orientations nominate near-duplicates by pigeonhole bucketing; every
  candidate is then **verified on decoded pixels**, so each reported pair carries a measured
  similarity score and matching orientation (`99.4% similar, rot90`) — never a hash coincidence.
- Train/val leakage, COCO structural defects (duplicate ids, dangling references, degenerate boxes
  and polygons, zero-annotation images), corrupt files. Exit code 1 on failure — drops straight
  into CI.
- Rotated, flipped, re-encoded and resized copies all caught. Pillow is the only dependency.


## 🌾 KrishiDisha — multi-task CV for agriculture, validated in the field

First-author **IEEE ICoEIT 2025** paper ([doi:10.1109/ICoEIT63558.2025.11211713](https://doi.org/10.1109/ICoEIT63558.2025.11211713),
pp. 1028–1040). One platform, three jobs off the same imagery:

```mermaid
graph LR
    A["Field imagery"] --> B["Shared vision backbone"]
    B --> C["Disease detection<br/>classification"]
    B --> D["Crop recommendation<br/>multi-class"]
    B --> E["Yield prediction<br/>regression"]
    C & D & E --> F["Advisory delivered<br/>to the farmer"]
```

- Classification **F1 0.99**, precision **0.96**; yield regression **R² 0.98**.
- The number that actually matters: **field-validated with 150+ farmers** — real users on real
  plots, not only a held-out split. Most agri-CV work stops at the test set; this one went
  outside and got used.
- Multi-task by design — detection, classification and regression share one backbone and one
  inference pass, which is what makes it deployable on the cheap hardware a farm advisory
  service can actually afford.

---

## ⚡ The toolbox

| | |
|---|---|
| **Quantization** | FP16 · INT8 · UINT8 · UINT4 — TensorRT, ONNX Runtime, TFLite, OpenVINO |
| **Edge targets** | NVIDIA Jetson (Nano / Xavier / Orin) · Android robots · custom embedded boards · GPU servers |
| **Perception** | YOLO family · SAM-1/2/3 · D-FINE · RF-DETR · Faster/Mask R-CNN · pose · depth · anomaly detection |
| **Robot learning** | MuJoCo Warp · mjlab · RSL-RL (PPO) · reward & curriculum design · sim-to-real thinking |
| **Core** | PyTorch · TensorFlow · OpenCV · MediaPipe · C++ · Python · classical CV & projective geometry |

**Public lab notebook** → [Kaggle](https://www.kaggle.com/shivpratap0007): 50+ notebooks, 19 datasets — D-FINE fine-tuning with ONNX/OpenVINO export, SlowFast action recognition, MoveNet+LSTM pose pipelines, face embedding, and the G1 locomotion runs above, all reproducible.

## 📄 Publications

- **KrishiDisha: Revolutionizing Agriculture with Intelligent Recommendations, Disease Detection, and Yield Prediction** — ***first author*** · *2025 IEEE International Conference on Engineering Innovations and Technologies (ICoEIT), pp. 1028–1040* · [doi:10.1109/ICoEIT63558.2025.11211713](https://doi.org/10.1109/ICoEIT63558.2025.11211713). Multi-task CV platform, **field-validated with 150+ farmers**.
- **Web-BCD: A Machine and Deep Learning Approach for Breast Cancer Detection** — *IBM Technical Report, Dec 2024*. 89%→94% accuracy (ROC-AUC 0.96), **deployed live for clinician use**.
- **A Review on Understanding the Patterns of Student Dropout** — ***first author*** · *Studies in Smart Technologies, Springer, pp. 235–250, 2025* · [doi:10.1007/978-981-97-9006-7_20](https://doi.org/10.1007/978-981-97-9006-7_20).

## 💼 The short version

**ML Engineer · Kody Technolab** (Nov 2025 – present; intern Apr–Oct 2025) · **DL Research Mentee · IBM India** (2024)
**B.Tech CSE (AI/ML)** · Medi-Caps University · CGPA 8.67

Figures, architecture diagrams and measured results → **[shivpratapsinghpanwar.github.io](https://shivpratapsinghpanwar.github.io)**

---

<div align="center">

*I do the boring analysis of research papers, design for compute-constrained reality, merge approaches, and ship the result.*

</div>
