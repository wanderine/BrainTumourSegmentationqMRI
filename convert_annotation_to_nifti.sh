#!/bin/bash

Subject=Subject23

# Subject17, so small annotations that it is difficult to know where it should be

#if [ -f "/home/andek67/Data/Gliom/${Subject}/Annotation.nii" ]; then 
#	echo "Skipping making annotation nifti"	
#else 
	dcm2niix_afni -o /home/andek67/Data/Gliom/${Subject} -f Annotation /home/andek67/Data/Gliom/${Subject}/Annotation
	rm /home/andek67/Data/Gliom/${Subject}/*.json
#fi

#rm /home/andek67/Data/Gliom/${Subject}/Annotationa.nii 

3drefit -orient RPI /home/andek67/Data/Gliom/${Subject}/Annotation.nii 

#flirt -in /home/andek67/Data/Gliom/${Subject}/T1FLAIR_GD.nii -ref /home/andek67/Data/Gliom/${Subject}/Annotation.nii -applyxfm -init identitymatrix.txt -out /home/andek67/Data/Gliom/${Subject}/T1FLAIR_GD_resampled.nii

flirt -interp nearestneighbour -in /home/andek67/Data/Gliom/${Subject}/Annotation.nii -ref /home/andek67/Data/Gliom/${Subject}/T1FLAIR_GD.nii -applyxfm -init translationmatrix_${Subject}.txt -out /home/andek67/Data/Gliom/${Subject}/Annotation_resampled.nii

#flirt -nosearch -dof 6 -interp nearestneighbour -in /home/andek67/Data/Gliom/${Subject}/Annotation.nii -ref /home/andek67/Data/Gliom/${Subject}/T1FLAIR_GD.nii -out /home/andek67/Data/Gliom/${Subject}/Annotation_resampled.nii 

