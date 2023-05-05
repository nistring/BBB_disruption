import os
import glob
import shutil
from tqdm import *
from subprocess import check_call
import dicom2nifti

def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return

def reorganize(input_dir='dicom_data', output_dir='FLAIR'):
    """Converts dicom data into nifti.

    Args:
        input_dir (str, optional): Defaults to 'data'.
        output_dir (str, optional): Defaults to 'FLAIR'.
    """
    create_dir(os.path.join('data', output_dir))
    for subject in os.listdir(input_dir):
        subject_dir = os.path.join(input_dir, subject)
        for i in os.listdir(subject_dir):
            create_dir(os.path.join('data', output_dir, i))
            dicom_dir = os.path.join(subject_dir, i)
            output_file = os.path.join('data', output_dir, i, subject+'.nii')
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_file, reorient_nifti=False)
            check_call(["gzip", output_file, "-f"])

# src_dir_list = ["ADNI1_Screening_AD", "ADNI1_Screening_NC",
#                 "ADNI2_Screening_AD", "ADNI2_Screening_NC"]
# dst_dir_list = ["AD", "NC", "AD", "NC"]

# parent_dir = os.path.dirname(os.getcwd())
# data_dir = os.path.join(parent_dir, "data")

# for in_dir, out_dir in zip(src_dir_list, dst_dir_list):
#     data_src_dir = os.path.join(data_dir, in_dir)
#     data_dst_dir = os.path.join(data_dir, "ADNI", out_dir)
#     create_dir(data_dst_dir)

#     print("Move files\nfrom: {0}\nto {1}".format(data_src_dir, data_dst_dir))

#     subjects = os.listdir(data_src_dir)
#     for subject in tqdm(subjects):
#         subj_dir = os.path.join(data_src_dir, subject)
#         src_path = glob.glob(os.path.join(subj_dir, "*", "*", "*", "*"))[0]
#         dst_path = os.path.join(data_dst_dir, subject + ".nii")
#         shutil.copyfile(src_path, dst_path)
#         check_call(["gzip", dst_path, "-f"])
