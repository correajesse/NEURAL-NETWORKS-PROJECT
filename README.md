# NEURAL-NETWORKS-PROJECT
README
1. Project Overview

This project develops a multimodal deep learning model to classify hand grasps using surface electromyography (EMG) and inertial measurement unit (IMU) signals from the forearm.  
The goal is to help control a prosthetic hand by recognising the intended grasp type (10 different grasps) in two scenarios:

- Within‑subject: training and testing on the same user (personalised model).
- Cross‑subject: training on a group of users and testing on a new, unseen user (generalised model).

The model is a 1D CNN that fuses EMG and IMU features, trained on the MeganePro dataset (NinaPro DB10).

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/1Z3IOM


2. Model Architecture

- EMG branch: 12 channels → 1D CNN (filters 16 → 32) → 32‑D feature vector.
- IMU branch: 36 channels → 1D CNN (filters 16 → 32) → 32‑D feature vector.
- Concatenation: 64‑D feature vector.
- Classifier: Fully connected layers (64 → 64 → 10) with ReLU, Dropout(0.6) and Softmax.

Regularisation:  
- Weight decay (1e-2)  
- Dropout (0.6)  
- Label smoothing (0.1)  
- Gradient clipping (norm 1.0)  
- MixUp (only for cross‑subject)  
- Early stopping  

---

3. Dataset

Source: [MeganePro dataset 1 (MDS1)](https://doi.org/10.7910/DVN/1Z3IOM) – NinaPro DB10.  
Subjects: 45 (30 intact, 15 amputees). After excluding invalid subjects (S024, S025, S039, S108, S115):  
- Intact: 28 subjects  
- Amputees: 13 subjects  
- Total used: 41 subjects  (4 were marked as invalid by the author dataset)

Signals:  
- EMG (12 channels, 2000 Hz)  
- IMU (36 channels, 2000 Hz)  
- (Gaze and video were excluded in the final pipeline)

Preprocessing (script `03_preprocess_clip.py`):  
- Band‑pass filtering (20–450 Hz) for EMG, low‑pass (10 Hz) for IMU  
- Rectification (absolute value) for EMG  
- Sliding window (400 ms, stride 100 ms) → subsampled to 200 samples (200 ms effective window)  
- Subject‑wise z‑score normalisation (using only training data)  
- Train/test split by repetitions (5,6 as test)

Data reduction: 50% random subsampling per subject to speed up training.

---

Pipeline

1. Raw EMG/IMU 
2. Preprocessing (filter, sliding window, normalisation) 
3. Data augmentation (time shift, channel permutation, noise, scaling; plus MixUp for cross‑subject) 
4. CNN feature extraction (separate branches for EMG and IMU) 
5. Concatenation  
6. Classifier
7. Prediction (grasp type 0–9)

---

Results

| Experiment | Accuracy (average) | F1 macro |
|------------|--------------------|-----------|
| Within‑subject (41 subjects) | 65.0% | 0.639 |
| Cross‑subject (32 train, 9 test) | 33.3% | 0.332 |

Best within‑subject performance: 89.1% (subject S034)  
Worst within‑subject performance: 42.9% (subject S113)  

The cross‑subject model improves over the EMG‑only baseline (~25%), demonstrating the benefit of IMU fusion, but generalisation remains challenging.

---

How to Run

Data was download from the source page, "tabular data" was selected and 9 zips with 10 files each were placed in the folder /home/jesse.correa*****/Downloads

## 1. Environment

Use the provided Singularity container (PyTorch 22.01 with CUDA).

```bash
singularity exec --nv /raid/images/pytorch_22.01-py3.sif bash
```

Install dependencies

```bash
pip install -r requirements.txt
```
## 2. Extract data 
extract data from the download folder with 00_extract_zips.py
Extract the 9 NinaPro DB10 .zip files and organize the resulting .mat files.
Run this FIRST, before any other script.

Expected input structure:

/path/to/zips/

├── S01.zip (or any .zip file)

├── S02.zip

Then run 01_explore_mat.py
First script: explore the contect of the files .mat of MeganeProDB10. Run to understand the structure of the .mat load_dataset.py

Loads and organizes all NinaPro DB10 (MeganePro) .mat files into a unified format ready for preprocessing.

Actual variable names in DB10 (different from other NinaPro versions):

emg → sEMG forearm (N, 12)

acc → accelerometers (N, 36) multiple sensors

grasp → grasp label (N,) equivalent to "stimulus"

regrasp → re-proc label (N,) is equivalent to "restimulus" 
grasprepetition → repetition (N,) 
regrasprepetition→ repetition re-proc.(N,) 
gazepoint → gaze x,y (N, 2) 
gazepoint_invalid→ invalid mask (N,) 
tobiiacc → IMU head acc (N, 3) 
tobiigyr → IMU head gyro (N, 3) 
object → grabbed object (N,) 
position → object position (N,) 
dynamic → dynamic movement(N,)

Use: 
from load_dataset import NinaProDB10Dataset 
ds = NinaProDB10Dataset("/path/to/data") 
data = ds.load_subject(subject_id=22)

## 3. Preprocess the data (only needed once)

```bash
python src/03_preprocess_clip.py --data_dir /path/to/raw/matfiles --out_dir ./processed --window_ms 400 --stride_ms 100 --modalities emg acc --exclude_rest
```

To explore the new .npz files use exploration.py (change the path inside your code)

Then convert to `.pt`:

```bash
python src/05_cache_tensors.py --input_dir ./processed --output_dir ./cache_sub4 --subsample 4
```

(Actual paths may differ; adjust according to your directory structure.)
If you want to analyze the final data you are going to use, generate graphs and statistics with 
    python3 analyze_data.py --cache_dir cache_pt --out_dir analysis/
    
### 4. Train all experiments (within & cross)

```bash
python src/modelo_vid.py
```

This will train within‑subject and cross‑subject models for all populations (intact, amputee, all) and both modalities (EMG only, EMG+IMU). Results, models, and confusion matrices are saved in `resultados_finales/`.

### 5. Evaluate a single subject with the cross‑subject model and compare the results through models 

```bash
python src/inference.py --subject S012
```

---

## Repository Structure

grasp_multimodal
├── src/
│   ├── 03_preprocess_clip.py       # Preprocessing (filtering, windowing, normalisation)
│   ├── 05_cache_tensors.py         # Convert .npz to .pt
│   ├── run_all_experiments_fixed.py # Train all within and cross experiments
│   ├── inference_demo.py            # Predict on a subject (cross‑subject model)
│   ├── compare_models.py            # Compare within‑subject vs cross‑subject
│   └── ...
├── resultados_finales/              # Results, models, confusion matrices
├── logs/                            # SLURM logs
├── requirements.txt
└── README.md
```

---

## License & Citation

This project uses the MeganePro dataset. Please cite the original paper:  
*Cognolato et al. (2020). Gaze, visual, myoelectric, and inertial data of grasps for intelligent prosthetics. Scientific Data, 7, 43.*

---

## Author

Jesse Correa – Yachay Tech University  
Instructor: Jonathan Cruz


