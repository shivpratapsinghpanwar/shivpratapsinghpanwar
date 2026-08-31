# Shivpratap Singh Panwar

**Computer Vision Research Engineer** — Robotics Perception · Humanoid RL · Edge Inference · Medical AI

Ahmedabad, India · [Email](mailto:shivpratapsinghpanwar19@gmail.com) · [LinkedIn](https://www.linkedin.com/in/shivpratap-singh-panwar/) · [Kaggle](https://www.kaggle.com/shivpratap0007) · [GitHub](https://github.com/shivpratapsinghpanwar)

I build perception and control systems that have to work in the real world: humanoid locomotion policies trained end to end, vision pipelines for medical robots where no prior CV solution exists, and models quantized down to INT4 so they run on the hardware that's actually in the room.

---

## 🤖 Now: end-to-end humanoid development

Training **whole-body locomotion policies for the Unitree G1 (23-DoF)** — the full loop from task design to a policy that survives being shoved:

- **Robust locomotion suite** — [my fork of unitree_rl_mjlab](https://github.com/shivpratapsinghpanwar/unitree_rl_mjlab_custom_robot) (Unitree Robotics' RL stack on [mjlab](https://github.com/mujocolab/mjlab), the Isaac-Lab-style API over MuJoCo Warp) with a custom task registry: stand / walk / run / omnidirectional push-and-drag recovery, trained with PPO (RSL-RL).
- **Speed-adaptive gait clock** — the gait cycle shortens with commanded speed and drops stance fraction below 0.5, opening a genuine **flight phase** so running is learnable rather than a fast walk; reverse walking is a first-class command range, not an afterthought.
- **Perturbation curriculum**: instantaneous velocity impulses *and* sustained horizontal drags from uniformly random headings, ramped in via curriculum — policies that take hits from any direction.
- **Training infrastructure that survives reality**: multi-GPU runs on free Kaggle T4s with continuous checkpointing and cross-session resume; a diagnose-and-fix loop with dedicated failure-analysis runs (e.g. isolating a backward-push failure mode before retraining).
- Exploring **hierarchical robot foundation models** — [custom design work](https://github.com/shivpratapsinghpanwar/tau-0-vla_Custom_Design) on [τ0-VLA](https://github.com/sii-research/tau-0-vla) (world-model-guided test-time computation for robot control).

## 🏥 Medical robotics perception — Kody Technolab

Building the vision stack for a **medical robot detecting rare pediatric anomalies** (microcephaly, hydrocephaly, clubfoot, cleft lip/palate) — conditions with minimal training data and **no existing CV solution** at development time. Low-data strategies, classical projective geometry as the fallback where deep learning runs out of data, and multi-task perception (detection, instance segmentation, pose, depth) across security, advertisement, and data-gathering robots plus the Mahindra Assistant platform.

## 🏭 [Synthetic Data Factory](https://github.com/shivpratapsinghpanwar/Synthetic_Data_Factory)

An autonomously operated pipeline that generates synthetic medical imagery and **measures whether it actually improves detectors** — not whether it looks nice:

- Stable Diffusion 1.5 + per-class LoRA and a **from-scratch DDPM** (no natural-image prior) as pluggable backends; every synthetic image provenance-tracked and screened against **memorization of real patient images** before it may train anything.
- Honest, paired multi-seed evaluation on HAM10000 skin lesions: rare-class augmentation moved vascular-lesion F1 **+0.050 ± 0.013** and melanoma recall **+0.116 ± 0.048** ([measured results](https://github.com/shivpratapsinghpanwar/Synthetic_Data_Factory/blob/main/docs/results.md)).
- Runs its whole train/evaluate loop remotely on free Kaggle GPUs via a git-pinned execution runner built for agent-driven iteration.

---

## ⚡ Edge inference

| | |
|---|---|
| **Quantization** | FP16 · INT8 · UINT8 · UINT4 — TensorRT, ONNX Runtime, TFLite, OpenVINO |
| **Targets** | NVIDIA Jetson (Nano / Xavier / Orin), Android robots, custom embedded boards, GPU servers |
| **Perception** | YOLO family, SAM-1/2/3, Faster/Mask R-CNN, pose & depth estimation, anomaly detection |
| **Stack** | PyTorch, TensorFlow, OpenCV, MediaPipe, C++, classical CV & projective geometry |

Shipped: CV pipelines over **500K+ images / 100+ hours of video**, 10+ detector benchmark (best **95% mAP**), ~30% production throughput gains from architecture + inference optimization.

My [Kaggle](https://www.kaggle.com/shivpratap0007) is the public lab notebook — **50+ notebooks, 19 datasets**: D-FINE fine-tuning with ONNX/OpenVINO export, SlowFast video action recognition, MoveNet+LSTM pose pipelines, MobileOne×ArcFace face embedding, RF-DETR detection, and the G1 locomotion training runs above.

---

## 📄 Publications

- **KrishiDisha: Revolutionizing Agriculture with Intelligent Recommendations using Computer Vision** — *IEEE ICoEIT, Jul 2025*. Multi-task CV platform (F1 0.99 / precision 0.96 / R² 0.98), field-validated by 150+ farmers.
- **Web-BCD: A Machine and Deep Learning Approach for Breast Cancer Detection** — *IBM Technical Report, Dec 2024*. 89%→94% accuracy (ROC-AUC 0.96), deployed live for clinician use.
- **Understanding the Patterns of Student Dropout: A Review** — *Springer, Smart Technology, Jun 2024* ([chapter](https://link.springer.com/chapter/10.1007/978-981-97-9006-7_20)).

## 💼 Experience

- **ML Engineer · Kody Technolab** — Nov 2025 – present (intern Apr–Oct 2025)
- **Deep Learning Research Mentee · IBM India** — Jul–Dec 2024

**B.Tech CSE (AI/ML)** · Medi-Caps University, Indore · 2021–2025 · CGPA 8.67 · Head of Research & Astronomy, Science Club

---

*Older projects (KrishiDisha app, Web-BCD, dropout prediction, world-suicide-data analysis) live in the repos below.*
