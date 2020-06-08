#!/bin/bash

#set +e

rm AllMasks/*

rm AllT1GD/*
rm AllT1/*
rm AllT2FLAIR/*
rm AllT2/*

rm AllqMRIT1/*
rm AllqMRIT2/*
rm AllqMRIPD/*

rm AllqMRIT1GD/*
rm AllqMRIT2GD/*
rm AllqMRIPDGD/*

rm AllWM/*
rm AllGM/*
rm AllCSF/*
rm AllNON/*

rm AllWMGD/*
rm AllGMGD/*
rm AllCSFGD/*
rm AllNONGD/*

rm AllADC/*

#set -e

Subjects=0

for SubjectNumber in 1 2 4 5 6 7 8 9 10 11 12 13 14 15 16 18 19 20 21 22; do

    # Check if subject has all files, otherwise skip
    if [ -f  "Subject${SubjectNumber}/qMRI_T1_reg.nii.gz" ]; then

		((Subjects++))

	    cp Subject${SubjectNumber}/Annotation_resampled.nii.gz AllMasks/Mask_${SubjectNumber}.nii.gz

		cp Subject${SubjectNumber}/T1FLAIR_GD.nii 		AllT1GD/T1FLAIR_GD_${SubjectNumber}.nii
		cp Subject${SubjectNumber}/T1FLAIR_reg.nii.gz 	AllT1/T1FLAIR_${SubjectNumber}.nii.gz	
		cp Subject${SubjectNumber}/T2FLAIR_reg.nii.gz 	AllT2FLAIR/T2FLAIR_${SubjectNumber}.nii.gz
		cp Subject${SubjectNumber}/T2_reg.nii.gz 		AllT2/T2_${SubjectNumber}.nii.gz

		cp Subject${SubjectNumber}/qMRI_T1_reg.nii.gz 	AllqMRIT1/qMRI_T1_${SubjectNumber}.nii.gz
		cp Subject${SubjectNumber}/qMRI_T2_reg.nii.gz 	AllqMRIT2/qMRI_T2_${SubjectNumber}.nii.gz
		cp Subject${SubjectNumber}/qMRI_PD_reg.nii.gz 	AllqMRIPD/qMRI_PD_${SubjectNumber}.nii.gz

        cp Subject${SubjectNumber}/qMRI_T1_GD_reg.nii.gz 	AllqMRIT1GD/qMRI_T1_GD_${SubjectNumber}.nii.gz
		cp Subject${SubjectNumber}/qMRI_T2_GD_reg.nii.gz 	AllqMRIT2GD/qMRI_T2_GD_${SubjectNumber}.nii.gz
		cp Subject${SubjectNumber}/qMRI_PD_GD_reg.nii.gz 	AllqMRIPDGD/qMRI_PD_GD_${SubjectNumber}.nii.gz

		cp Subject${SubjectNumber}/WM_reg.nii.gz 		AllWM/WM_${SubjectNumber}.nii.gz
		cp Subject${SubjectNumber}/GM_reg.nii.gz 		AllGM/GM_${SubjectNumber}.nii.gz
		cp Subject${SubjectNumber}/CSF_reg.nii.gz 		AllCSF/CSF_${SubjectNumber}.nii.gz
		cp Subject${SubjectNumber}/NON_reg.nii.gz 		AllNON/NON_${SubjectNumber}.nii.gz

		cp Subject${SubjectNumber}/WM_GD_reg.nii.gz 		AllWMGD/WM_GD_${SubjectNumber}.nii.gz
		cp Subject${SubjectNumber}/GM_GD_reg.nii.gz 		AllGMGD/GM_GD_${SubjectNumber}.nii.gz
		cp Subject${SubjectNumber}/CSF_GD_reg.nii.gz 		AllCSFGD/CSF_GD_${SubjectNumber}.nii.gz
		cp Subject${SubjectNumber}/NON_GD_reg.nii.gz 		AllNONGD/NON_GD_${SubjectNumber}.nii.gz

		cp Subject${SubjectNumber}/ADC_reg.nii.gz 		AllADC/ADC_${SubjectNumber}.nii.gz

		gzip AllT1GD/T1FLAIR_GD_${SubjectNumber}.nii

    	if [ $SubjectNumber -lt 4 ]; then
        	echo "Removing slices for subject $SubjectNumber"

			3dZeropad -prefix AllMasks/temp_mask.nii.gz -I -6 -S -6 AllMasks/Mask_${SubjectNumber}.nii.gz 
			rm AllMasks/Mask_${SubjectNumber}.nii.gz 
			mv AllMasks/temp_mask.nii.gz AllMasks/Mask_${SubjectNumber}.nii.gz

        	3dZeropad -prefix AllT1GD/temp_MRI.nii.gz  		-I -6 -S -6 AllT1GD/T1FLAIR_GD_${SubjectNumber}.nii.gz 
			3dZeropad -prefix AllT1/temp_MRI.nii.gz  		-I -6 -S -6 AllT1/T1FLAIR_${SubjectNumber}.nii.gz
        	3dZeropad -prefix AllT2FLAIR/temp_MRI.nii.gz  	-I -6 -S -6 AllT2FLAIR/T2FLAIR_${SubjectNumber}.nii.gz
			3dZeropad -prefix AllT2/temp_MRI.nii.gz  		-I -6 -S -6 AllT2/T2_${SubjectNumber}.nii.gz

        	3dZeropad -prefix AllqMRIT1/temp_MRI.nii.gz  	-I -6 -S -6 AllqMRIT1/qMRI_T1_${SubjectNumber}.nii.gz 
			3dZeropad -prefix AllqMRIT2/temp_MRI.nii.gz  	-I -6 -S -6 AllqMRIT2/qMRI_T2_${SubjectNumber}.nii.gz
			3dZeropad -prefix AllqMRIPD/temp_MRI.nii.gz  	-I -6 -S -6 AllqMRIPD/qMRI_PD_${SubjectNumber}.nii.gz

        	3dZeropad -prefix AllqMRIT1GD/temp_MRI.nii.gz  	-I -6 -S -6 AllqMRIT1GD/qMRI_T1_GD_${SubjectNumber}.nii.gz 
			3dZeropad -prefix AllqMRIT2GD/temp_MRI.nii.gz  	-I -6 -S -6 AllqMRIT2GD/qMRI_T2_GD_${SubjectNumber}.nii.gz
			3dZeropad -prefix AllqMRIPDGD/temp_MRI.nii.gz  	-I -6 -S -6 AllqMRIPDGD/qMRI_PD_GD_${SubjectNumber}.nii.gz

			3dZeropad -prefix AllWM/temp_MRI.nii.gz  	-I -6 -S -6 AllWM/WM_${SubjectNumber}.nii.gz
			3dZeropad -prefix AllGM/temp_MRI.nii.gz  	-I -6 -S -6 AllGM/GM_${SubjectNumber}.nii.gz
			3dZeropad -prefix AllCSF/temp_MRI.nii.gz  	-I -6 -S -6 AllCSF/CSF_${SubjectNumber}.nii.gz
			3dZeropad -prefix AllNON/temp_MRI.nii.gz  	-I -6 -S -6 AllNON/NON_${SubjectNumber}.nii.gz

			3dZeropad -prefix AllWMGD/temp_MRI.nii.gz  	    -I -6 -S -6 AllWMGD/WM_GD_${SubjectNumber}.nii.gz
			3dZeropad -prefix AllGMGD/temp_MRI.nii.gz  	    -I -6 -S -6 AllGMGD/GM_GD_${SubjectNumber}.nii.gz
			3dZeropad -prefix AllCSFGD/temp_MRI.nii.gz  	-I -6 -S -6 AllCSFGD/CSF_GD_${SubjectNumber}.nii.gz
			3dZeropad -prefix AllNONGD/temp_MRI.nii.gz  	-I -6 -S -6 AllNONGD/NON_GD_${SubjectNumber}.nii.gz

			3dZeropad -prefix AllADC/temp_MRI.nii.gz  	-I -6 -S -6 AllADC/ADC_${SubjectNumber}.nii.gz

			rm AllT1GD/T1FLAIR_GD_${SubjectNumber}.nii.gz 
			rm AllT1/T1FLAIR_${SubjectNumber}.nii.gz 
			rm AllT2FLAIR/T2FLAIR_${SubjectNumber}.nii.gz 
			rm AllT2/T2_${SubjectNumber}.nii.gz 
			rm AllqMRIT1/qMRI_T1_${SubjectNumber}.nii.gz 
			rm AllqMRIT2/qMRI_T2_${SubjectNumber}.nii.gz 
			rm AllqMRIPD/qMRI_PD_${SubjectNumber}.nii.gz 
			rm AllqMRIT1GD/qMRI_T1_GD_${SubjectNumber}.nii.gz 
			rm AllqMRIT2GD/qMRI_T2_GD_${SubjectNumber}.nii.gz 
			rm AllqMRIPDGD/qMRI_PD_GD_${SubjectNumber}.nii.gz 
			rm AllWM/WM_${SubjectNumber}.nii.gz 
			rm AllGM/GM_${SubjectNumber}.nii.gz 
			rm AllCSF/CSF_${SubjectNumber}.nii.gz 
			rm AllNON/NON_${SubjectNumber}.nii.gz 
			rm AllWMGD/WM_GD_${SubjectNumber}.nii.gz 
			rm AllGMGD/GM_GD_${SubjectNumber}.nii.gz 
			rm AllCSFGD/CSF_GD_${SubjectNumber}.nii.gz 
			rm AllNONGD/NON_GD_${SubjectNumber}.nii.gz 
			rm AllADC/ADC_${SubjectNumber}.nii.gz 

			mv AllT1GD/temp_MRI.nii.gz 		AllT1GD/T1FLAIR_GD_${SubjectNumber}.nii.gz 
			mv AllT1/temp_MRI.nii.gz 		AllT1/T1FLAIR_${SubjectNumber}.nii.gz
			mv AllT2FLAIR/temp_MRI.nii.gz 	AllT2FLAIR/T2FLAIR_${SubjectNumber}.nii.gz
			mv AllT2/temp_MRI.nii.gz 		AllT2/T2_${SubjectNumber}.nii.gz
			mv AllqMRIT1/temp_MRI.nii.gz	AllqMRIT1/qMRI_T1_${SubjectNumber}.nii.gz 
			mv AllqMRIT2/temp_MRI.nii.gz	AllqMRIT2/qMRI_T2_${SubjectNumber}.nii.gz
			mv AllqMRIPD/temp_MRI.nii.gz	AllqMRIPD/qMRI_PD_${SubjectNumber}.nii.gz
			mv AllqMRIT1GD/temp_MRI.nii.gz	AllqMRIT1GD/qMRI_T1_GD_${SubjectNumber}.nii.gz 
			mv AllqMRIT2GD/temp_MRI.nii.gz	AllqMRIT2GD/qMRI_T2_GD_${SubjectNumber}.nii.gz
			mv AllqMRIPDGD/temp_MRI.nii.gz	AllqMRIPDGD/qMRI_PD_GD_${SubjectNumber}.nii.gz
			mv AllWM/temp_MRI.nii.gz		AllWM/WM_${SubjectNumber}.nii.gz
			mv AllGM/temp_MRI.nii.gz		AllGM/GM_${SubjectNumber}.nii.gz
			mv AllCSF/temp_MRI.nii.gz		AllCSF/CSF_${SubjectNumber}.nii.gz
			mv AllNON/temp_MRI.nii.gz		AllNON/NON_${SubjectNumber}.nii.gz
            mv AllWMGD/temp_MRI.nii.gz		AllWMGD/WM_GD_${SubjectNumber}.nii.gz
			mv AllGMGD/temp_MRI.nii.gz		AllGMGD/GM_GD_${SubjectNumber}.nii.gz
			mv AllCSFGD/temp_MRI.nii.gz		AllCSFGD/CSF_GD_${SubjectNumber}.nii.gz
			mv AllNONGD/temp_MRI.nii.gz		AllNONGD/NON_GD_${SubjectNumber}.nii.gz
			mv AllADC/temp_MRI.nii.gz		AllADC/ADC_${SubjectNumber}.nii.gz

    	fi
	fi

done

echo "Copied data from $Subjects subjects"



