# Quantative BBB disruption
A disruption of Brain-blood-barrier is suspected for predicting the prognosis of patients, yet there haven't been many studies in terms of quantitative measurements of it.
In general, brain parenchymas are often stained with Gd enhancement dye. This repo tries to validate the usefulness of T2 FLAIR enhanced image in calculating amount of enhancement and its associations with clinical outcomes.

This work relies on preprocessing software by [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL) and [SPM](https://www.fil.ion.ucl.ac.uk/spm/)

## Start
In order to fully take the advantage of this work, it is needed to install ``FSL`` and ``SPM`` with ``MatLab``.
A helpful snippet of code is provided as
```
mkdir BBB
cd BBB
git clone https://github.com/nistring/BBB_disruption.git
conda create -n BBB python==3.10
conda activate BBB
# Essential lib
pip install fslpy dcm2niix nibabel nilearn
# Set fsl path
export FSLDIR=path/to/fsl
source $FSLDIR/etc/fslconf/fsl.sh
```