import pydicom
import nibabel as nib
import dicom2nifti
import os

dicom_dir = r"\\sbmdcl_nas\dicomer$\orig_data\PE_CT\4015004745696"
output_file = "../data/4015004745696.nii"
dicom2nifti.convert_directory(dicom_dir,os.path.dirname(output_file),compression=False)