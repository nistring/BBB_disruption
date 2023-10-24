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
pip install fslpy dcm2niix nibabel
export FSLDIR=path/to/fsl
source $FSLDIR/etc/fslconf/fsl.sh
```