# Quantative BBB disruption

<img width="2000" height="909" alt="image" src="https://github.com/user-attachments/assets/aa858b6b-c022-4458-96a9-588fcbbaf53f" />

A disruption of Brain-blood-barrier is suspected for predicting the prognosis of patients, yet there haven't been many studies in terms of quantitative measurements of it.
In general, brain parenchymas are often stained with Gd enhancement dye. This repo tries to validate the usefulness of T2 FLAIR enhanced image in calculating amount of enhancement and its associations with clinical outcomes.

## Methodology
To quantify BBB disruption, pre-contrast and post-contrast FLAIR MRI images were processed to calculate the percentage volume (PV) of enhancement:

### Brain Tissue Extraction
T1-weighted images were processed using [sMRIPrep](https://github.com/nipreps/smriprep) to remove non-brain tissues (CSF, cranium, soft tissue).
Gray and white matter masks were used to isolate brain tissue.

### Image Alignment & Normalization
Post-contrast images were aligned to pre-contrast images using [FLIRT](https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/FLIRT.html).
Signal intensities were normalized (divided by mean values), then subtracted to highlight contrast enhancement.
Subtracted images were further normalized (by standard deviation) and resized to 512×512×24 pixels.

### Thresholding & Filtering
Enhancement maps were thresholded from 0.5 to 5.0 (in 0.5 intervals).
Flood fill algorithm suppressed outer surface enhancements.
Pixels below the mean intensity in post-contrast images were removed to exclude extra-axial artifacts.

### PV Calculation
PV was defined as the percentage of enhanced voxels within brain tissue relative to the total brain tissue voxels.
