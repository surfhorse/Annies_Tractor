# Third-party models and software

afv itself contains **no AI model and no third-party code**. Its install scripts (`install/`) download
the models and tools below from their publishers onto your computer. Each keeps its own licence, and
by installing and using them you accept those terms. This page records what each project stated when
it was checked on **2026-11-09**. It is not legal advice; read the linked terms yourself, especially
before using results commercially.

Copies of the licence texts as they were on that date are in `licenses/third_party/`. The exact
versions afv was tested with (commits and weight hashes) are in `install/tested_versions/`.
`afv models --check` prints this register for the stages you have installed.

## AI models

**Summary: on the path afv uses, every model allows commercial use of its outputs.** Two carry
conditions you must know about:

- **GEM-X** weights are under the **NVIDIA Open Model License**, which has its own conditions (for
  example, it can be revoked if the model's guardrails are circumvented). Read it on the model card.
- **GeoCalib** weights are **CC BY 4.0**: if you publish something made with afv, **credit GeoCalib**
  (for example in your game's acknowledgements).

| Model | Used for | Code licence | Weight licence | Where the terms are | Commercial use of outputs |
| --- | --- | --- | --- | --- | --- |
| [GEM-X](https://github.com/NVlabs/GEM-X) body, motion, `vitpose.pth` | 3D pose and motion | Apache-2.0 (some files OpenMDW-1.1) | NVIDIA Open Model License | [model card `nvidia/GEM-X`](https://huggingface.co/nvidia/GEM-X) | Yes, under the NVIDIA OML's conditions |
| [SOMA-X](https://huggingface.co/nvidia/SOMA-X) v0.3.3 assets | Body model | Apache-2.0 | Apache-2.0 | model card | Yes |
| [RF-DETR](https://github.com/roboflow/rf-detr) Medium | Finding people | Apache-2.0 | Apache-2.0 | README (sizes N, S, M, L are Apache-2.0; XL, 2XL and the "plus" sizes are not: afv does not use them) | Yes |
| [SAM 2.1](https://github.com/facebookresearch/sam2) Hiera-L | Tracking the fighters (masks) | Apache-2.0 | Apache-2.0 | repo licence covers the checkpoints | Yes |
| [ViPE](https://github.com/nv-tlabs/vipe) | Camera solve | Apache-2.0 | (none of its own) | repo licence | Yes |
| [GeoCalib](https://github.com/cvg/GeoCalib) pinhole (via ViPE) | Initial focal length | Apache-2.0 | **CC BY 4.0** | README "License" | Yes, **with attribution** |
| [MoGe-2](https://huggingface.co/Ruicheng/moge-2-vitl) ViT-L (via ViPE) | Depth | MIT (DINOv2 parts Apache-2.0) | MIT | model card | Yes |
| [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) `droid.pth` (via ViPE) | Camera tracking | BSD-3-Clause | none stated; repo licence assumed | repo licence | Yes (assumed) |
| [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) Swin-T (via ViPE) | Boxes for masking | Apache-2.0 | none stated; repo licence assumed | repo licence | Yes (assumed) |
| bert-base-uncased (via ViPE) | GroundingDINO's text encoder | Apache-2.0 | Apache-2.0 | model card | Yes |
| [SAM](https://github.com/facebookresearch/segment-anything) ViT-B (via ViPE) | Masks for the camera solve | Apache-2.0 | Apache-2.0 | repo licence covers the model | Yes |
| [DeAOT](https://github.com/yoxu515/aot-benchmark) R50 (via ViPE) | Mask propagation | BSD-3-Clause | none stated; repo licence assumed | repo licence | Yes (assumed) |
| YOLOX-X HumanArt ONNX (OpenMMLab) | GEM-X's own person detector | Apache-2.0 | **not established** | — | It is downloaded and loaded by GEM-X, but afv gives GEM-X its own boxes, so this model's detections are not used |

"Assumed" means the project publishes its weights from the same repository and states no separate
weight licence, so the repository licence is taken to cover them. If that matters to you, ask the
authors.

Not reviewed: the **training data** behind each model (for example SA-1B for SAM, COCO for RF-DETR,
GEM-X's motion-capture data). If you plan to sell something made with afv, review those terms too.

ViPE can use other components that are non-commercial (MetricTrail, Unik3D). The pipeline afv runs
(`pose_only_long`) does not use them.

## Other software the installers download

| Software | Licence | Note |
| --- | --- | --- |
| [uv](https://github.com/astral-sh/uv) | Apache-2.0 or MIT | Python environment manager |
| FFmpeg static build (johnvansickle.com) | GPL-3.0 build | Used as a separate program to read and cut video |
| [Miniforge](https://github.com/conda-forge/miniforge) and conda-forge packages | BSD-3-Clause; per package | Supplies the CUDA 12.8 compiler for ViPE |
| PyTorch (CUDA 12.8 wheels) | BSD-3-Clause; bundled NVIDIA CUDA libraries under NVIDIA's licence | |
| numpy, scipy (BSD-3-Clause), pydantic, typer (MIT), PyAV (BSD-3-Clause; its wheels bundle FFmpeg libraries) | as stated | afv's own Python dependencies |

[Blender](https://www.blender.org) (GPL) is installed by you; afv drives it and the add-on runs
inside it, which is why everything under `blender/` is GPL-3.0-or-later.

## Your footage

afv works on video you supply. Fight broadcasts and films are usually copyrighted, and the people in
them have their own rights. afv's licence gives you no rights to any footage, and motion extracted from
someone else's footage may be restricted by their rights. Use footage you own or are allowed to use.
