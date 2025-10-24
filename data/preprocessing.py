import os
import numpy as np
from multiprocessing import cpu_count, Pool
from pathlib import Path
import shutil
from nipype.interfaces.dcm2nii import Dcm2niix


def convert():
    # Set the base directory containing DICOM files
    dicom_base_dir = Path("dicom")

    # Create a Dcm2niix object
    converter = Dcm2niix()

    # Iterate over all DICOM directories
    for dicom_dir in dicom_base_dir.rglob("*"):
        if dicom_dir.name in ['diffusion']:#['e', 'ne', 't1', 't2', 'diffusion']:
            output_dir = Path("nii") / dicom_dir.relative_to(dicom_base_dir)  # Use the nii directory for output


            # Create the output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)

            # Set the source and output directories for the converter
            converter.inputs.compress = 'n'
            converter.inputs.source_dir = dicom_dir
            converter.inputs.output_dir = output_dir
            converter.inputs.out_filename = "_".join(output_dir.parts[-3:])

            # Run the conversion
            converter.run()

def rename():
    base_dir = Path("t1t2")

    for dirpath in base_dir.rglob("*"):
        if dirpath.is_dir() and " " in dirpath.name:
            new_name = dirpath.name.replace(" ", "_")
            new_path = dirpath.with_name(new_name)
            dirpath.rename(new_path)

def move():
    ene_base_dir = Path("ene")
    t1t2_base_dir = Path("t1t2")
    dst_base_dir = Path("dicom")

    for t1t2_dir in t1t2_base_dir.iterdir():
        for seq in t1t2_dir.iterdir():
            if seq.name[0] == "1":
                ene_dir = ene_base_dir / t1t2_dir.name
                if not ene_dir.is_dir():
                    continue
            elif seq.name[0] == "2":
                ene_dir = ene_base_dir / str(int(t1t2_dir.name))
                if not ene_dir.is_dir():
                    continue
            else:
                raise ValueError("Unexpected sequence name")

            dst_dir = dst_base_dir / seq.relative_to(t1t2_base_dir)
            
            for dicom_file in ene_dir.rglob("*.dcm"):
                if dicom_file.parent.name in ["e", "ne"]:
                    relative_path = dicom_file.relative_to(dicom_file.parent.parent)
                    destination_path = dst_dir / relative_path.parent.name.lower() / relative_path.name
                    destination_path.parent.mkdir(parents=True, exist_ok=True)
                    dicom_file.rename(destination_path)
                if "dw" in dicom_file.parent.name:
                    relative_path = dicom_file.relative_to(dicom_file.parent.parent)
                    destination_path = dst_dir / "diffusion" / relative_path.parent.name.lower() / relative_path.name
                    destination_path.parent.mkdir(parents=True, exist_ok=True)
                    dicom_file.rename(destination_path)

            #print(destination_path, dicom_file)
            for dicom_file in seq.rglob("*.dcm"):
                relative_path = dicom_file.relative_to(dicom_file.parent.parent)
                destination_path = dst_dir / relative_path.parent.name.lower() / relative_path.name
                destination_path.parent.mkdir(parents=False, exist_ok=True)
                
                dicom_file.rename(destination_path)
            #print(destination_path, dicom_file)

def check():
    nii_base_dir = Path("nii")
    nii_files = list(nii_base_dir.rglob("*.nii"))
    cnt = 0 
    for nii_dir in nii_base_dir.iterdir():
        for nii_file in nii_dir.iterdir():
            cnt += 1
    print(cnt)
    print(f"Number of .nii files: {len(nii_files)}")

def prepare_spm():
    nii_base_dir = Path("nii")
    spm_base_dir = Path("spm_input")

    for seq in ['t1', 't2']:
        (spm_base_dir / seq).mkdir(parents=True, exist_ok=True)
        for nii_file in nii_base_dir.rglob(f"*{seq}*.nii"):
            dst_path = spm_base_dir / seq / nii_file.name
            shutil.copy(nii_file, dst_path)


from fsl.wrappers import flirt

def FLIRT(subject):
    subject_dir = f"nii/{subject}"
    src_file = f"{subject_dir}/e/{subject.replace('/','_') + '_e.nii'}"
    ref_file = f"{subject_dir}/ne/{subject.replace('/','_') + '_ne.nii'}"
    dst_file = f"{subject_dir}/e_align/{subject.replace('/','_') + '_e_align.nii'}"
    if not os.path.exists(dst_file):
        Path(dst_file).parent.mkdir(parents=True, exist_ok=True)
        flirt(
            src_file,
            ref_file,
            out=dst_file,
            twod=True,
        )
        print(dst_file, " done")

def to_BIDS():
    nii_base_dir = Path("nii")
    bids_base_dir = Path("bids")

    for seq in ['t1', 't2']:
        for nii_file in nii_base_dir.rglob(f"*{seq}*.nii"):
            subject = nii_file.name.split("_")[0]
            session = nii_file.name.split("_")[1]
            session = "first" if session[0] == "1" else "second"
            dst_dir = bids_base_dir / f"sub-{subject}{session}" / "anat"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_file = dst_dir / f"sub-{subject}{session}_{seq.upper()}w.nii"
            shutil.copy(nii_file, dst_file)
            shutil.copy(nii_file.with_suffix(".json"), dst_file.with_suffix(".json"))

# rename()
# move()
# convert()
# to_BIDS()

# if __name__ == "__main__":
#     processes = cpu_count()

#     subjects = []
#     for x in os.listdir("spm_out/gm"):
#         x = x.replace("c1", "").replace("_t1.nii", "").split("_")
#         subjects.append(x[0] + '/' + '_'.join(x[1:]))
#     subjects = np.array_split(subjects, processes)
#     subjects = [list(s) for s in subjects]

#     def do(*sub):
#         for s in sub:
#             FLIRT(s)

#     with Pool(processes=processes) as pool:
#         pool.starmap(do, subjects)
