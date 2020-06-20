#!/bin/bash

#set +e

rm Skullstrip/AllT1GD/*
rm Skullstrip/AllT1/*
rm Skullstrip/AllT2FLAIR/*
rm Skullstrip/AllT2/*

rm Skullstrip/AllqMRIT1/*
rm Skullstrip/AllqMRIT2/*
rm Skullstrip/AllqMRIPD/*

rm Skullstrip/AllqMRIT1GD/*
rm Skullstrip/AllqMRIT2GD/*
rm Skullstrip/AllqMRIPDGD/*

rm Skullstrip/AllNONGD/*

#set -e

Subjects=0

for SubjectNumber in 1 2 4 5 6 7 8 9 10 11 12 13 14 15 16 18 19 20 21 22; do

    # Check if subject has all files, otherwise skip
    if [ -f  "Subject${SubjectNumber}/qMRI_T1_reg.nii.gz" ]; then

		((Subjects++))

		cp Subject${SubjectNumber}/T1FLAIR_GD.nii 		Skullstrip/AllT1GD/T1FLAIR_GD_${SubjectNumber}.nii
		cp Subject${SubjectNumber}/T1FLAIR_reg.nii.gz 	Skullstrip/AllT1/T1FLAIR_${SubjectNumber}.nii.gz	
		cp Subject${SubjectNumber}/T2FLAIR_reg.nii.gz 	Skullstrip/AllT2FLAIR/T2FLAIR_${SubjectNumber}.nii.gz
		cp Subject${SubjectNumber}/T2_reg.nii.gz 		Skullstrip/AllT2/T2_${SubjectNumber}.nii.gz

		cp Subject${SubjectNumber}/qMRI_T1_reg.nii.gz 	Skullstrip/AllqMRIT1/qMRI_T1_${SubjectNumber}.nii.gz
		cp Subject${SubjectNumber}/qMRI_T2_reg.nii.gz 	Skullstrip/AllqMRIT2/qMRI_T2_${SubjectNumber}.nii.gz
		cp Subject${SubjectNumber}/qMRI_PD_reg.nii.gz 	Skullstrip/AllqMRIPD/qMRI_PD_${SubjectNumber}.nii.gz

        cp Subject${SubjectNumber}/qMRI_T1_GD_reg.nii.gz 	Skullstrip/AllqMRIT1GD/qMRI_T1_GD_${SubjectNumber}.nii.gz
		cp Subject${SubjectNumber}/qMRI_T2_GD_reg.nii.gz 	Skullstrip/AllqMRIT2GD/qMRI_T2_GD_${SubjectNumber}.nii.gz
		cp Subject${SubjectNumber}/qMRI_PD_GD_reg.nii.gz 	Skullstrip/AllqMRIPDGD/qMRI_PD_GD_${SubjectNumber}.nii.gz

		cp Subject${SubjectNumber}/NON_GD_reg.nii.gz 		Skullstrip/AllNONGD/NON_GD_${SubjectNumber}.nii.gz

		cp Subject${SubjectNumber}/brainmask.nii.gz 		Skullstrip/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz

		gzip Skullstrip/AllT1GD/T1FLAIR_GD_${SubjectNumber}.nii

    	if [ $SubjectNumber -lt 4 ]; then
        	echo "Removing slices for subject $SubjectNumber"

			
        	3dZeropad -prefix Skullstrip/AllT1GD/temp_MRI.nii.gz  		-I -6 -S -6 Skullstrip/AllT1GD/T1FLAIR_GD_${SubjectNumber}.nii.gz 
			3dZeropad -prefix Skullstrip/AllT1/temp_MRI.nii.gz  		-I -6 -S -6 Skullstrip/AllT1/T1FLAIR_${SubjectNumber}.nii.gz
        	3dZeropad -prefix Skullstrip/AllT2FLAIR/temp_MRI.nii.gz  	-I -6 -S -6 Skullstrip/AllT2FLAIR/T2FLAIR_${SubjectNumber}.nii.gz
			3dZeropad -prefix Skullstrip/AllT2/temp_MRI.nii.gz  		-I -6 -S -6 Skullstrip/AllT2/T2_${SubjectNumber}.nii.gz

        	3dZeropad -prefix Skullstrip/AllqMRIT1/temp_MRI.nii.gz  	-I -6 -S -6 Skullstrip/AllqMRIT1/qMRI_T1_${SubjectNumber}.nii.gz 
			3dZeropad -prefix Skullstrip/AllqMRIT2/temp_MRI.nii.gz  	-I -6 -S -6 Skullstrip/AllqMRIT2/qMRI_T2_${SubjectNumber}.nii.gz
			3dZeropad -prefix Skullstrip/AllqMRIPD/temp_MRI.nii.gz  	-I -6 -S -6 Skullstrip/AllqMRIPD/qMRI_PD_${SubjectNumber}.nii.gz

        	3dZeropad -prefix Skullstrip/AllqMRIT1GD/temp_MRI.nii.gz  	-I -6 -S -6 Skullstrip/AllqMRIT1GD/qMRI_T1_GD_${SubjectNumber}.nii.gz 
			3dZeropad -prefix Skullstrip/AllqMRIT2GD/temp_MRI.nii.gz  	-I -6 -S -6 Skullstrip/AllqMRIT2GD/qMRI_T2_GD_${SubjectNumber}.nii.gz
			3dZeropad -prefix Skullstrip/AllqMRIPDGD/temp_MRI.nii.gz  	-I -6 -S -6 Skullstrip/AllqMRIPDGD/qMRI_PD_GD_${SubjectNumber}.nii.gz

			3dZeropad -prefix Skullstrip/AllNONGD/temp_MRI.nii.gz  	-I -6 -S -6 Skullstrip/AllNONGD/NON_GD_${SubjectNumber}.nii.gz

			3dZeropad -prefix Skullstrip/AllBrainmasks/temp_MRI.nii.gz  	-I -6 -S -6 Skullstrip/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz

			rm Skullstrip/AllT1GD/T1FLAIR_GD_${SubjectNumber}.nii.gz 
			rm Skullstrip/AllT1/T1FLAIR_${SubjectNumber}.nii.gz 
			rm Skullstrip/AllT2FLAIR/T2FLAIR_${SubjectNumber}.nii.gz 
			rm Skullstrip/AllT2/T2_${SubjectNumber}.nii.gz 
			rm Skullstrip/AllqMRIT1/qMRI_T1_${SubjectNumber}.nii.gz 
			rm Skullstrip/AllqMRIT2/qMRI_T2_${SubjectNumber}.nii.gz 
			rm Skullstrip/AllqMRIPD/qMRI_PD_${SubjectNumber}.nii.gz 
			rm Skullstrip/AllqMRIT1GD/qMRI_T1_GD_${SubjectNumber}.nii.gz 
			rm Skullstrip/AllqMRIT2GD/qMRI_T2_GD_${SubjectNumber}.nii.gz 
			rm Skullstrip/AllqMRIPDGD/qMRI_PD_GD_${SubjectNumber}.nii.gz 
			rm Skullstrip/AllNONGD/NON_GD_${SubjectNumber}.nii.gz 
			rm Skullstrip/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz 

			mv Skullstrip/AllT1GD/temp_MRI.nii.gz 		Skullstrip/AllT1GD/T1FLAIR_GD_${SubjectNumber}.nii.gz 
			mv Skullstrip/AllT1/temp_MRI.nii.gz 		Skullstrip/AllT1/T1FLAIR_${SubjectNumber}.nii.gz
			mv Skullstrip/AllT2FLAIR/temp_MRI.nii.gz 	Skullstrip/AllT2FLAIR/T2FLAIR_${SubjectNumber}.nii.gz
			mv Skullstrip/AllT2/temp_MRI.nii.gz 		Skullstrip/AllT2/T2_${SubjectNumber}.nii.gz
			mv Skullstrip/AllqMRIT1/temp_MRI.nii.gz		Skullstrip/AllqMRIT1/qMRI_T1_${SubjectNumber}.nii.gz 
			mv Skullstrip/AllqMRIT2/temp_MRI.nii.gz		Skullstrip/AllqMRIT2/qMRI_T2_${SubjectNumber}.nii.gz
			mv Skullstrip/AllqMRIPD/temp_MRI.nii.gz		Skullstrip/AllqMRIPD/qMRI_PD_${SubjectNumber}.nii.gz
			mv Skullstrip/AllqMRIT1GD/temp_MRI.nii.gz	Skullstrip/AllqMRIT1GD/qMRI_T1_GD_${SubjectNumber}.nii.gz 
			mv Skullstrip/AllqMRIT2GD/temp_MRI.nii.gz	Skullstrip/AllqMRIT2GD/qMRI_T2_GD_${SubjectNumber}.nii.gz
			mv Skullstrip/AllqMRIPDGD/temp_MRI.nii.gz	Skullstrip/AllqMRIPDGD/qMRI_PD_GD_${SubjectNumber}.nii.gz
			mv Skullstrip/AllNONGD/temp_MRI.nii.gz		Skullstrip/AllNONGD/NON_GD_${SubjectNumber}.nii.gz
			mv Skullstrip/AllBrainmasks/temp_MRI.nii.gz		Skullstrip/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz

    	fi

        # Apply brain mask
        fslmaths Skullstrip/AllT1GD/T1FLAIR_GD_${SubjectNumber}.nii.gz     -mul Skullstrip/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz Skullstrip/AllT1GD/T1FLAIR_GD_${SubjectNumber}.nii.gz
        fslmaths Skullstrip/AllT1/T1FLAIR_${SubjectNumber}.nii.gz          -mul Skullstrip/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz Skullstrip/AllT1/T1FLAIR_${SubjectNumber}.nii.gz
        fslmaths Skullstrip/AllT2FLAIR/T2FLAIR_${SubjectNumber}.nii.gz     -mul Skullstrip/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz Skullstrip/AllT2FLAIR/T2FLAIR_${SubjectNumber}.nii.gz
        fslmaths Skullstrip/AllT2/T2_${SubjectNumber}.nii.gz               -mul Skullstrip/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz Skullstrip/AllT2/T2_${SubjectNumber}.nii.gz
        fslmaths Skullstrip/AllqMRIT1/qMRI_T1_${SubjectNumber}.nii.gz      -mul Skullstrip/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz Skullstrip/AllqMRIT1/qMRI_T1_${SubjectNumber}.nii.gz
        fslmaths Skullstrip/AllqMRIT2/qMRI_T2_${SubjectNumber}.nii.gz      -mul Skullstrip/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz Skullstrip/AllqMRIT2/qMRI_T2_${SubjectNumber}.nii.gz
        fslmaths Skullstrip/AllqMRIPD/qMRI_PD_${SubjectNumber}.nii.gz      -mul Skullstrip/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz Skullstrip/AllqMRIPD/qMRI_PD_${SubjectNumber}.nii.gz
        fslmaths Skullstrip/AllqMRIT1GD/qMRI_T1_GD_${SubjectNumber}.nii.gz -mul Skullstrip/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz Skullstrip/AllqMRIT1GD/qMRI_T1_GD_${SubjectNumber}.nii.gz
        fslmaths Skullstrip/AllqMRIT2GD/qMRI_T2_GD_${SubjectNumber}.nii.gz -mul Skullstrip/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz Skullstrip/AllqMRIT2GD/qMRI_T2_GD_${SubjectNumber}.nii.gz
        fslmaths Skullstrip/AllqMRIPDGD/qMRI_PD_GD_${SubjectNumber}.nii.gz -mul Skullstrip/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz Skullstrip/AllqMRIPDGD/qMRI_PD_GD_${SubjectNumber}.nii.gz
        fslmaths Skullstrip/AllNONGD/NON_GD_${SubjectNumber}.nii.gz        -mul Skullstrip/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz Skullstrip/AllNONGD/NON_GD_${SubjectNumber}.nii.gz

	fi

done

echo "Copied data from $Subjects subjects"



