# Brain_FLAIR
This repo visualizes effect of contrast-enhanced brain MRI.
This work relies on preprocessing software by [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL) and its user-friendly application of [quqixun's work](https://github.com/quqixun/BrainPrep)

## Start
Follow installation steps of https://github.com/quqixun/BrainPrep
```
mkdir Brain_FLAIR
cd Brain_FLAIR
git clone https://github.com/nistring/Brain_FLAIR.git
conda create -n FLAIR python==3.10
conda activate FLAIR
pip install -r requirements.txt
```
Place your DICOM files in `BrainPrep/dicom_data`

## Preprocessing
```
cd BrainPrep
python preprocessing.py
```
The `preprocessing.py` uses `src` package and is a basic skeleton for making preprocessing pipeline.
It is designed to strip off skull from MRI and leaves only brain parenchyme for further processing.

## Run
```
python main.py
```

## Results
Structural similarity between w/ contrast and w/o contrast MRI images. Large discrepencies are highlighted in red.