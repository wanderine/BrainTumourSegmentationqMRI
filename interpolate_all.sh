#!/bin/bash

# Copy original data for first 2 subjects, since they have different slice thichkness

for SubjectNumber in 1 2; do

    cp Subject${SubjectNumber}/T1FLAIR_GD.nii 		SkullstripTemp/AllT1GD/T1FLAIR_GD_${SubjectNumber}.nii
	cp Subject${SubjectNumber}/T1FLAIR_reg.nii.gz 	SkullstripTemp/AllT1/T1FLAIR_${SubjectNumber}.nii.gz	
	cp Subject${SubjectNumber}/T2FLAIR_reg.nii.gz 	SkullstripTemp/AllT2FLAIR/T2FLAIR_${SubjectNumber}.nii.gz
	cp Subject${SubjectNumber}/T2_reg.nii.gz 		SkullstripTemp/AllT2/T2_${SubjectNumber}.nii.gz

	cp Subject${SubjectNumber}/qMRI_T1_reg.nii.gz 	SkullstripTemp/AllqMRIT1/qMRI_T1_${SubjectNumber}.nii.gz
	cp Subject${SubjectNumber}/qMRI_T2_reg.nii.gz 	SkullstripTemp/AllqMRIT2/qMRI_T2_${SubjectNumber}.nii.gz
	cp Subject${SubjectNumber}/qMRI_PD_reg.nii.gz 	SkullstripTemp/AllqMRIPD/qMRI_PD_${SubjectNumber}.nii.gz

    cp Subject${SubjectNumber}/qMRI_T1_GD_reg.nii.gz 	SkullstripTemp/AllqMRIT1GD/qMRI_T1_GD_${SubjectNumber}.nii.gz
	cp Subject${SubjectNumber}/qMRI_T2_GD_reg.nii.gz 	SkullstripTemp/AllqMRIT2GD/qMRI_T2_GD_${SubjectNumber}.nii.gz
	cp Subject${SubjectNumber}/qMRI_PD_GD_reg.nii.gz 	SkullstripTemp/AllqMRIPDGD/qMRI_PD_GD_${SubjectNumber}.nii.gz

	cp Subject${SubjectNumber}/NON_GD_reg.nii.gz 		SkullstripTemp/AllNONGD/NON_GD_${SubjectNumber}.nii.gz

	cp Subject${SubjectNumber}/brainmask.nii.gz 		SkullstripTemp/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz

	cp Subject${SubjectNumber}/Annotation_resampled.nii.gz 		SkullstripTemp/AllMasks/Mask_${SubjectNumber}.nii.gz

	gzip SkullstripTemp/AllT1GD/T1FLAIR_GD_${SubjectNumber}.nii

    # Apply brain mask
    fslmaths SkullstripTemp/AllT1GD/T1FLAIR_GD_${SubjectNumber}.nii.gz     -mul SkullstripTemp/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz SkullstripTemp/AllT1GD/T1FLAIR_GD_${SubjectNumber}.nii.gz
    fslmaths SkullstripTemp/AllT1/T1FLAIR_${SubjectNumber}.nii.gz          -mul SkullstripTemp/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz SkullstripTemp/AllT1/T1FLAIR_${SubjectNumber}.nii.gz
    fslmaths SkullstripTemp/AllT2FLAIR/T2FLAIR_${SubjectNumber}.nii.gz     -mul SkullstripTemp/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz SkullstripTemp/AllT2FLAIR/T2FLAIR_${SubjectNumber}.nii.gz
    fslmaths SkullstripTemp/AllT2/T2_${SubjectNumber}.nii.gz               -mul SkullstripTemp/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz SkullstripTemp/AllT2/T2_${SubjectNumber}.nii.gz
    fslmaths SkullstripTemp/AllqMRIT1/qMRI_T1_${SubjectNumber}.nii.gz      -mul SkullstripTemp/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz SkullstripTemp/AllqMRIT1/qMRI_T1_${SubjectNumber}.nii.gz
    fslmaths SkullstripTemp/AllqMRIT2/qMRI_T2_${SubjectNumber}.nii.gz      -mul SkullstripTemp/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz SkullstripTemp/AllqMRIT2/qMRI_T2_${SubjectNumber}.nii.gz
    fslmaths SkullstripTemp/AllqMRIPD/qMRI_PD_${SubjectNumber}.nii.gz      -mul SkullstripTemp/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz SkullstripTemp/AllqMRIPD/qMRI_PD_${SubjectNumber}.nii.gz
    fslmaths SkullstripTemp/AllqMRIT1GD/qMRI_T1_GD_${SubjectNumber}.nii.gz -mul SkullstripTemp/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz SkullstripTemp/AllqMRIT1GD/qMRI_T1_GD_${SubjectNumber}.nii.gz
    fslmaths SkullstripTemp/AllqMRIT2GD/qMRI_T2_GD_${SubjectNumber}.nii.gz -mul SkullstripTemp/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz SkullstripTemp/AllqMRIT2GD/qMRI_T2_GD_${SubjectNumber}.nii.gz
    fslmaths SkullstripTemp/AllqMRIPDGD/qMRI_PD_GD_${SubjectNumber}.nii.gz -mul SkullstripTemp/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz SkullstripTemp/AllqMRIPDGD/qMRI_PD_GD_${SubjectNumber}.nii.gz
    fslmaths SkullstripTemp/AllNONGD/NON_GD_${SubjectNumber}.nii.gz        -mul SkullstripTemp/AllBrainmasks/brainmask_${SubjectNumber}.nii.gz SkullstripTemp/AllNONGD/NON_GD_${SubjectNumber}.nii.gz

done

for Data in 1 2 3 4 5 6 7 8 9 10 11; do

    if [ "$Data" -eq "1" ] ; then
        DataDirectory=AllNONGD
        FileName=NON_GD
    elif [ "$Data" -eq "2" ] ; then
        DataDirectory=AllqMRIPD
        FileName=qMRI_PD
    elif [ "$Data" -eq "3" ] ; then
        DataDirectory=AllqMRIPDGD
        FileName=qMRI_PD_GD
    elif [ "$Data" -eq "4" ] ; then
        DataDirectory=AllqMRIT1
        FileName=qMRI_T1
    elif [ "$Data" -eq "5" ] ; then
        DataDirectory=AllqMRIT1GD
        FileName=qMRI_T1_GD
    elif [ "$Data" -eq "6" ] ; then
        DataDirectory=AllqMRIT2
        FileName=qMRI_T2
    elif [ "$Data" -eq "7" ] ; then
        DataDirectory=AllqMRIT2GD
        FileName=qMRI_T2_GD
    elif [ "$Data" -eq "8" ] ; then
        DataDirectory=AllT1
        FileName=T1FLAIR
    elif [ "$Data" -eq "9" ] ; then
        DataDirectory=AllT1GD
        FileName=T1FLAIR_GD
    elif [ "$Data" -eq "10" ] ; then
        DataDirectory=AllT2
        FileName=T2
    elif [ "$Data" -eq "11" ] ; then
        DataDirectory=AllT2FLAIR
        FileName=T2FLAIR
    fi

    echo "Interpolating $FileName" 

    for SubjectNumber in 1 2; do

        /usr/local/fsl/bin/fslcreatehd 512 512 79 1 0.4297 0.4297 2 1 0 0 0 16  /home/andek67/Data/Gliom/SkullstripInterpolated/${DataDirectory}/${FileName}_${SubjectNumber}_tmp.nii.gz ; /usr/local/fsl/bin/flirt -in /home/andek67/Data/Gliom/SkullstripTemp/${DataDirectory}/${FileName}_${SubjectNumber}.nii.gz -applyxfm -init /usr/local/fsl/etc/flirtsch/ident.mat -out /home/andek67/Data/Gliom/SkullstripInterpolated/${DataDirectory}/${FileName}_${SubjectNumber} -paddingsize 0.0 -interp sinc -sincwidth 7 -sincwindow hanning -datatype float -ref /home/andek67/Data/Gliom/SkullstripInterpolated/${DataDirectory}/${FileName}_${SubjectNumber}_tmp &

    done

    for SubjectNumber in 4 5 6 7 8 9 10 11 12 13 14 15 16 18 19 20 21 22; do

        /usr/local/fsl/bin/fslcreatehd 512 512 72 1 0.4297 0.4297 2 1 0 0 0 16  /home/andek67/Data/Gliom/SkullstripInterpolated/${DataDirectory}/${FileName}_${SubjectNumber}_tmp.nii.gz ; /usr/local/fsl/bin/flirt -in /home/andek67/Data/Gliom/Skullstrip/${DataDirectory}/${FileName}_${SubjectNumber}.nii.gz -applyxfm -init /usr/local/fsl/etc/flirtsch/ident.mat -out /home/andek67/Data/Gliom/SkullstripInterpolated/${DataDirectory}/${FileName}_${SubjectNumber} -paddingsize 0.0 -interp sinc -sincwidth 7 -sincwindow hanning -datatype float -ref /home/andek67/Data/Gliom/SkullstripInterpolated/${DataDirectory}/${FileName}_${SubjectNumber}_tmp &

    done

    wait 

    rm /home/andek67/Data/Gliom/SkullstripInterpolated/${DataDirectory}/*tmp*

done

#------------

# Now interpolate the masks with NN interpolation

DataDirectory=AllMasks
DataDirectoryOut=AllMasksInterpolated
FileName=Mask

echo "Interpolating masks"

for SubjectNumber in 1 2; do

    /usr/local/fsl/bin/fslcreatehd 512 512 79 1 0.4297 0.4297 2 1 0 0 0 16  /home/andek67/Data/Gliom/${DataDirectoryOut}/${FileName}_${SubjectNumber}_tmp.nii.gz ; /usr/local/fsl/bin/flirt -in /home/andek67/Data/Gliom/SkullstripTemp/${DataDirectory}/${FileName}_${SubjectNumber}.nii.gz -applyxfm -init /usr/local/fsl/etc/flirtsch/ident.mat -out /home/andek67/Data/Gliom/${DataDirectoryOut}/${FileName}_${SubjectNumber} -paddingsize 0.0 -interp nearestneighbour -datatype float -ref /home/andek67/Data/Gliom/${DataDirectoryOut}/${FileName}_${SubjectNumber}_tmp &

done

for SubjectNumber in 4 5 6 7 8 9 10 11 12 13 14 15 16 18 19 20 21 22; do

    /usr/local/fsl/bin/fslcreatehd 512 512 72 1 0.4297 0.4297 2 1 0 0 0 16  /home/andek67/Data/Gliom/${DataDirectoryOut}/${FileName}_${SubjectNumber}_tmp.nii.gz ; /usr/local/fsl/bin/flirt -in /home/andek67/Data/Gliom/${DataDirectory}/${FileName}_${SubjectNumber}.nii.gz -applyxfm -init /usr/local/fsl/etc/flirtsch/ident.mat -out /home/andek67/Data/Gliom/${DataDirectoryOut}/${FileName}_${SubjectNumber} -paddingsize 0.0 -interp nearestneighbour -datatype float -ref /home/andek67/Data/Gliom/${DataDirectoryOut}/${FileName}_${SubjectNumber}_tmp &

done

wait 

rm /home/andek67/Data/Gliom/${DataDirectoryOut}/*tmp*

#-------------
# Now remove 7 extra slices for 2 first subjects

for SubjectNumber in 1 2; do

	3dZeropad -prefix SkullstripInterpolated/AllT1GD/temp_MRI.nii.gz  		-I -3 -S -4 SkullstripInterpolated/AllT1GD/T1FLAIR_GD_${SubjectNumber}.nii.gz 
	3dZeropad -prefix SkullstripInterpolated/AllT1/temp_MRI.nii.gz  		-I -3 -S -4 SkullstripInterpolated/AllT1/T1FLAIR_${SubjectNumber}.nii.gz
    3dZeropad -prefix SkullstripInterpolated/AllT2FLAIR/temp_MRI.nii.gz  	-I -3 -S -4 SkullstripInterpolated/AllT2FLAIR/T2FLAIR_${SubjectNumber}.nii.gz
    3dZeropad -prefix SkullstripInterpolated/AllT2/temp_MRI.nii.gz  		-I -3 -S -4 SkullstripInterpolated/AllT2/T2_${SubjectNumber}.nii.gz

    3dZeropad -prefix SkullstripInterpolated/AllqMRIT1/temp_MRI.nii.gz  	-I -3 -S -4 SkullstripInterpolated/AllqMRIT1/qMRI_T1_${SubjectNumber}.nii.gz 
    3dZeropad -prefix SkullstripInterpolated/AllqMRIT2/temp_MRI.nii.gz  	-I -3 -S -4 SkullstripInterpolated/AllqMRIT2/qMRI_T2_${SubjectNumber}.nii.gz
	3dZeropad -prefix SkullstripInterpolated/AllqMRIPD/temp_MRI.nii.gz  	-I -3 -S -4 SkullstripInterpolated/AllqMRIPD/qMRI_PD_${SubjectNumber}.nii.gz

    3dZeropad -prefix SkullstripInterpolated/AllqMRIT1GD/temp_MRI.nii.gz  	-I -3 -S -4 SkullstripInterpolated/AllqMRIT1GD/qMRI_T1_GD_${SubjectNumber}.nii.gz 
    3dZeropad -prefix SkullstripInterpolated/AllqMRIT2GD/temp_MRI.nii.gz  	-I -3 -S -4 SkullstripInterpolated/AllqMRIT2GD/qMRI_T2_GD_${SubjectNumber}.nii.gz
	3dZeropad -prefix SkullstripInterpolated/AllqMRIPDGD/temp_MRI.nii.gz  	-I -3 -S -4 SkullstripInterpolated/AllqMRIPDGD/qMRI_PD_GD_${SubjectNumber}.nii.gz

	3dZeropad -prefix SkullstripInterpolated/AllNONGD/temp_MRI.nii.gz     	-I -3 -S -4 SkullstripInterpolated/AllNONGD/NON_GD_${SubjectNumber}.nii.gz

	3dZeropad -prefix AllMasksInterpolated/temp_mask.nii.gz     	        -I -3 -S -4 AllMasksInterpolated/Mask_${SubjectNumber}.nii.gz


	rm SkullstripInterpolated/AllT1GD/T1FLAIR_GD_${SubjectNumber}.nii.gz 
	rm SkullstripInterpolated/AllT1/T1FLAIR_${SubjectNumber}.nii.gz 
	rm SkullstripInterpolated/AllT2FLAIR/T2FLAIR_${SubjectNumber}.nii.gz 
	rm SkullstripInterpolated/AllT2/T2_${SubjectNumber}.nii.gz 
	rm SkullstripInterpolated/AllqMRIT1/qMRI_T1_${SubjectNumber}.nii.gz 
	rm SkullstripInterpolated/AllqMRIT2/qMRI_T2_${SubjectNumber}.nii.gz 
	rm SkullstripInterpolated/AllqMRIPD/qMRI_PD_${SubjectNumber}.nii.gz 
	rm SkullstripInterpolated/AllqMRIT1GD/qMRI_T1_GD_${SubjectNumber}.nii.gz 
	rm SkullstripInterpolated/AllqMRIT2GD/qMRI_T2_GD_${SubjectNumber}.nii.gz 
	rm SkullstripInterpolated/AllqMRIPDGD/qMRI_PD_GD_${SubjectNumber}.nii.gz 
	rm SkullstripInterpolated/AllNONGD/NON_GD_${SubjectNumber}.nii.gz 
    rm AllMasksInterpolated/Mask_${SubjectNumber}.nii.gz

	mv SkullstripInterpolated/AllT1GD/temp_MRI.nii.gz 		SkullstripInterpolated/AllT1GD/T1FLAIR_GD_${SubjectNumber}.nii.gz 
	mv SkullstripInterpolated/AllT1/temp_MRI.nii.gz 		SkullstripInterpolated/AllT1/T1FLAIR_${SubjectNumber}.nii.gz
	mv SkullstripInterpolated/AllT2FLAIR/temp_MRI.nii.gz 	SkullstripInterpolated/AllT2FLAIR/T2FLAIR_${SubjectNumber}.nii.gz
	mv SkullstripInterpolated/AllT2/temp_MRI.nii.gz 		SkullstripInterpolated/AllT2/T2_${SubjectNumber}.nii.gz
	mv SkullstripInterpolated/AllqMRIT1/temp_MRI.nii.gz		SkullstripInterpolated/AllqMRIT1/qMRI_T1_${SubjectNumber}.nii.gz 
	mv SkullstripInterpolated/AllqMRIT2/temp_MRI.nii.gz		SkullstripInterpolated/AllqMRIT2/qMRI_T2_${SubjectNumber}.nii.gz
	mv SkullstripInterpolated/AllqMRIPD/temp_MRI.nii.gz		SkullstripInterpolated/AllqMRIPD/qMRI_PD_${SubjectNumber}.nii.gz
	mv SkullstripInterpolated/AllqMRIT1GD/temp_MRI.nii.gz	SkullstripInterpolated/AllqMRIT1GD/qMRI_T1_GD_${SubjectNumber}.nii.gz 
	mv SkullstripInterpolated/AllqMRIT2GD/temp_MRI.nii.gz	SkullstripInterpolated/AllqMRIT2GD/qMRI_T2_GD_${SubjectNumber}.nii.gz
	mv SkullstripInterpolated/AllqMRIPDGD/temp_MRI.nii.gz	SkullstripInterpolated/AllqMRIPDGD/qMRI_PD_GD_${SubjectNumber}.nii.gz
	mv SkullstripInterpolated/AllNONGD/temp_MRI.nii.gz		SkullstripInterpolated/AllNONGD/NON_GD_${SubjectNumber}.nii.gz
    mv AllMasksInterpolated/temp_mask.nii.gz				AllMasksInterpolated/Mask_${SubjectNumber}.nii.gz
			
done




