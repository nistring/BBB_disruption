# Brain_FLAIR
This repo visualizes effect of contrast-enhanced brain MRI

## Start
```
mkdir Brain_FLAIR
cd Brain_FLAIR
git clone https://github.com/nistring/Brain_FLAIR.git
conda create -n FLAIR python==3.10
pip install -r requirements.txt
```
Place your DICOM files in `data/`

## Run
```
python main.py
```

## Results
Structural similarity between w/ contrast and w/o contrast MRI images. Large discrepencies are highlighted in red.

![difference](https://user-images.githubusercontent.com/71208448/235330647-8c317055-7897-47e5-9250-fc8215bbbdf1.png)
