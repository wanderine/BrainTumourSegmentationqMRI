#!/bin/bash

for SubjectNumber in 12 14 21; do
#for SubjectNumber in 1 2 4 5 6 7 8 9 10 11 12 13 14 15 16 18 19 20 21 22 23; do

	# Register 

	flirt -coarsesearch 3 -finesearch 1 -dof 6 -searchrx -5 5 -searchry -5 5 -searchrz -5 5 -interp sinc -in Subject${SubjectNumber}/T1FLAIR.nii  -ref Subject${SubjectNumber}/T1FLAIR_GD.nii -out Subject${SubjectNumber}/T1FLAIR_reg.nii &

	#flirt -coarsesearch 3 -finesearch 1 -dof 6 -searchrx -25 25 -searchry -25 25 -searchrz -25 25 -interp sinc -in Subject${SubjectNumber}/ADC.nii      -ref Subject${SubjectNumber}/T1FLAIR_GD.nii -out Subject${SubjectNumber}/ADC_reg.nii &

	flirt -coarsesearch 3 -finesearch 1 -dof 6 -searchrx -5 5 -searchry -5 5 -searchrz -5 5 -interp sinc -in Subject${SubjectNumber}/T2FLAIR.nii  -ref Subject${SubjectNumber}/T1FLAIR_GD.nii -out Subject${SubjectNumber}/T2FLAIR_reg.nii &

	flirt -coarsesearch 3 -finesearch 1 -dof 6 -searchrx -5 5 -searchry -5 5 -searchrz -5 5 -interp sinc -in Subject${SubjectNumber}/T2.nii       -ref Subject${SubjectNumber}/T1FLAIR_GD.nii -out Subject${SubjectNumber}/T2_reg.nii &

    # Use PD to register qMRI volumes, as PD is most similar to T1FLAIR_GD
	flirt -coarsesearch 3 -finesearch 1 -dof 6 -searchrx -5 5 -searchry -5 5 -searchrz -5 5 -interp sinc -in Subject${SubjectNumber}/qMRI_PD.nii    -ref Subject${SubjectNumber}/T1FLAIR_GD.nii -out Subject${SubjectNumber}/qMRI_PD_reg.nii -omat Subject${SubjectNumber}/qMRI_PD_to_T1FLAIR_GD_Subject${SubjectNumber}.mat &

	flirt -coarsesearch 3 -finesearch 1 -dof 6 -searchrx -5 5 -searchry -5 5 -searchrz -5 5 -interp sinc -in Subject${SubjectNumber}/qMRI_PD_GD.nii    -ref Subject${SubjectNumber}/T1FLAIR_GD.nii -out Subject${SubjectNumber}/qMRI_PD_GD_reg.nii -omat Subject${SubjectNumber}/qMRI_PD_GD_to_T1FLAIR_GD_Subject${SubjectNumber}.mat &

	wait

	# Apply the same registration to all other qMRI volumes
	flirt -interp sinc -in Subject${SubjectNumber}/qMRI_T1.nii    -ref Subject${SubjectNumber}/T1FLAIR_GD.nii -applyxfm -init Subject${SubjectNumber}/qMRI_PD_to_T1FLAIR_GD_Subject${SubjectNumber}.mat -out Subject${SubjectNumber}/qMRI_T1_reg.nii &

	flirt -interp sinc -in Subject${SubjectNumber}/qMRI_T2.nii    -ref Subject${SubjectNumber}/T1FLAIR_GD.nii -applyxfm -init Subject${SubjectNumber}/qMRI_PD_to_T1FLAIR_GD_Subject${SubjectNumber}.mat -out Subject${SubjectNumber}/qMRI_T2_reg.nii &

	flirt -interp sinc -in Subject${SubjectNumber}/NON.nii    -ref Subject${SubjectNumber}/T1FLAIR_GD.nii -applyxfm -init Subject${SubjectNumber}/qMRI_PD_to_T1FLAIR_GD_Subject${SubjectNumber}.mat -out Subject${SubjectNumber}/NON_reg.nii &

	flirt -interp sinc -in Subject${SubjectNumber}/CSF.nii    -ref Subject${SubjectNumber}/T1FLAIR_GD.nii -applyxfm -init Subject${SubjectNumber}/qMRI_PD_to_T1FLAIR_GD_Subject${SubjectNumber}.mat -out Subject${SubjectNumber}/CSF_reg.nii &

	flirt -interp sinc -in Subject${SubjectNumber}/WM.nii    -ref Subject${SubjectNumber}/T1FLAIR_GD.nii -applyxfm -init Subject${SubjectNumber}/qMRI_PD_to_T1FLAIR_GD_Subject${SubjectNumber}.mat -out Subject${SubjectNumber}/WM_reg.nii &

	flirt -interp sinc -in Subject${SubjectNumber}/GM.nii    -ref Subject${SubjectNumber}/T1FLAIR_GD.nii -applyxfm -init Subject${SubjectNumber}/qMRI_PD_to_T1FLAIR_GD_Subject${SubjectNumber}.mat -out Subject${SubjectNumber}/GM_reg.nii &

	#---

	flirt -interp sinc -in Subject${SubjectNumber}/qMRI_T1_GD.nii    -ref Subject${SubjectNumber}/T1FLAIR_GD.nii -applyxfm -init Subject${SubjectNumber}/qMRI_PD_GD_to_T1FLAIR_GD_Subject${SubjectNumber}.mat -out Subject${SubjectNumber}/qMRI_T1_GD_reg.nii &

	flirt -interp sinc -in Subject${SubjectNumber}/qMRI_T2_GD.nii    -ref Subject${SubjectNumber}/T1FLAIR_GD.nii -applyxfm -init Subject${SubjectNumber}/qMRI_PD_GD_to_T1FLAIR_GD_Subject${SubjectNumber}.mat -out Subject${SubjectNumber}/qMRI_T2_GD_reg.nii &

	flirt -interp sinc -in Subject${SubjectNumber}/NON_GD.nii    -ref Subject${SubjectNumber}/T1FLAIR_GD.nii -applyxfm -init Subject${SubjectNumber}/qMRI_PD_GD_to_T1FLAIR_GD_Subject${SubjectNumber}.mat -out Subject${SubjectNumber}/NON_GD_reg.nii &

	flirt -interp sinc -in Subject${SubjectNumber}/CSF_GD.nii    -ref Subject${SubjectNumber}/T1FLAIR_GD.nii -applyxfm -init Subject${SubjectNumber}/qMRI_PD_GD_to_T1FLAIR_GD_Subject${SubjectNumber}.mat -out Subject${SubjectNumber}/CSF_GD_reg.nii &

	flirt -interp sinc -in Subject${SubjectNumber}/WM_GD.nii    -ref Subject${SubjectNumber}/T1FLAIR_GD.nii -applyxfm -init Subject${SubjectNumber}/qMRI_PD_GD_to_T1FLAIR_GD_Subject${SubjectNumber}.mat -out Subject${SubjectNumber}/WM_GD_reg.nii &

	flirt -interp sinc -in Subject${SubjectNumber}/GM_GD.nii    -ref Subject${SubjectNumber}/T1FLAIR_GD.nii -applyxfm -init Subject${SubjectNumber}/qMRI_PD_GD_to_T1FLAIR_GD_Subject${SubjectNumber}.mat -out Subject${SubjectNumber}/GM_GD_reg.nii &

	wait

done

