# Patient-disjoint benchmark of deep image features and mask-derived morphometrics on BoMBR

Code and derived data for *"Patient-Disjoint Evaluation of Deep Image Features and Mask-Derived Morphometrics for
Field-Level Reticulin Grading of Bone Marrow Trephine Images"* (submitted to Medical Engineering & Physics).

The BoMBR images and masks are **not** redistributed. Obtain them from the source authors (Raina et al., MELBA 2025,
doi:10.59275/j.melba.2025-171e) under their data-use agreement.

## Contents
| Folder | Content |
|---|---|
| `original_pipeline/` | Baseline pipeline: training scripts (`scripts/01–09`), loss implementation (`src/models.py`: 0.4 weighted CE + 0.4 soft Dice + 0.2 focal, γ = 2), original descriptor code (`src/data.py`), configuration, patient-disjoint split manifest and provenance. |
| `revision_analysis/` | All analyses reported in the paper (see the table below), the corrected descriptor definitions (`features.py`, `descriptor_defs.json`) and the mask-encoding unit test (`test_mask_encoding.py`). |
| `derived_data/` | Embeddings of all encoders and protocols, descriptor tables (reference and predicted masks), segmentation metrics and per-field Dice, every model result with test predictions (`results/res_*.json`), and the baseline run's prediction and descriptor tables. |
| `figures_source/` | Scripts that regenerate every figure from `derived_data`. |

## Important corrections (see manuscript Sections 2.4, 2.5, 2.7)
* **Mask encoding.** BoMBR masks are indexed **0 = Reticulin, 1 = Fat, 2 = Cell, 3 = Bone**. The baseline configuration used 0 = Bone and
  3 = Reticulin. `test_mask_encoding.py <Myelofibrozis_PNG root>` verifies the mapping from colour statistics.
* **Elongation.** The minor-axis floor is now 1 pixel (was 1e-6, which produced elongations of about 3e6 for collinear components).
* **Pixel scale.** Development fields are resampled by 0.65 to the pixel scale of the 512 × 512 test crops (`features.DEV_TO_TEST_SCALE`).

## Expected layout for re-running (inside `revision_analysis/`)
```
data/      -> BoMBR root (train/images, train/masks, test/images, test/masks, master_metadata.csv)
ckpt/      -> baseline checkpoints (segmentation_unetpp_resnet34_best.pt, image_grade_resnet50_best.pt)
weights/   -> dinov2_vits14_pretrain.pth, phikon_model.safetensors (owkin/phikon), resnet18-5c106cde.pth
```
Python 3.11, torch 2.5.1, torchvision 0.20.1, segmentation-models-pytorch 0.5, timm 1.0, scikit-learn 1.8, scikit-image 0.26, OpenCV 4.13.

## Order of execution
| Step | Command | Manuscript |
|---|---|---|
| 1 | `python test_mask_encoding.py data`; `python scale_from_masks.py` | Sections 2.4–2.5, Note S1 |
| 2 | `python seg_infer.py original test`, `... reinhard test`; `python finetune_seg.py 600`; `python seg_infer.py ft test`, `... ft dev` | Section 2.6 |
| 3 | `python segmetrics.py original reinhard ft`; `python seg_summary.py original reinhard ft` | Table 4, Figure 3, Table S6 |
| 4 | `python make_features.py gt`; `... gt legacyscale`; `... pred_original legacyscale`; `... pred_ft`; `... pred_reinhard` | Table 3, Table S1 |
| 5 | `python agreement.py pred_original pred_reinhard pred_ft` | Table 5, Table S2 |
| 6 | `python emb.py <encoder> <original|matched|matchedcn>` for resnet50_ft, resnet18_in, dinov2_vits14, phikon | Section 2.8 |
| 7 | `python exp_repro.py`, `exp_desc.py`, `exp_emb.py`, `exp_emb_cn.py`, `exp_emb_orig.py`, `exp_final.py`, `exp_cross.py`, `exp_fusion.py <pred_ft|gt> <encoder_protocol>` | Tables 6–8, S3–S5, S7 |
| 8 | `python collect.py; python values.py`, then the scripts in `figures_source/` | Figures 4–8 |

All model selection uses grouped cross-validation over the 19 development patients; the 50-field locked test set is used only for the
final prediction of each model. Seeds are fixed (20260812 split; 20260929 bootstrap and fine-tuning).

## Licence
MIT (code). Derived tables are provided for research reproducibility and remain subject to the BoMBR data-use conditions.
