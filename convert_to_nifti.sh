#!/bin/bash

Subject=Subject6

# Loop over directories
for i in /home/andek67/Data/Gliom/${Subject}/SCANS/*; do

	echo "This is directory $i"	

	file=`ls ${i}/DICOM/*.dcm | head -1`
	description=`dicom_hdr ${file} | grep '0008 103e'`
    
    #echo "Description is $description "

	if [[ $description == *"T1 FLAIR med GD"* ]]; then
	    echo "Found T1 GD"
        dcm2niix_afni -o /home/andek67/Data/Gliom/${Subject} -f T1FLAIR_GD ${i}
	elif [[ $description == *"T1 FLAIR"* ]]; then
	    echo "Found T1 FLAIR"
        dcm2niix_afni -o /home/andek67/Data/Gliom/${Subject} -f T1FLAIR ${i}
	elif [[ $description == *"T2 FLAIR"* ]]; then
	    echo "Found T2 FLAIR"
        dcm2niix_afni -o /home/andek67/Data/Gliom/${Subject} -f T2FLAIR ${i}
	elif [[ $description == *"T2 PROPELLER"* ]]; then
	    echo "Found T2 PROPELLER"
        dcm2niix_afni -o /home/andek67/Data/Gliom/${Subject} -f T2 ${i}
	elif [[ $description == *"Exponential Apparent"* ]]; then
	    echo "Found exponential apparent diffusion, doing nothing"
	elif [[ $description == *"Apparent Diffusion"* ]]; then
	    echo "Found apparent diffusion"
        dcm2niix_afni -o /home/andek67/Data/Gliom/${Subject} -f ADC ${i}
	elif [[ $description == *"Ax BRAVO"* ]]; then
	    echo "Found Ax BRAVO"
        dcm2niix_afni -o /home/andek67/Data/Gliom/${Subject} -f BRAVO ${i}
	fi	

	#rm /home/andek67/Data/Gliom/${Subject}/*.json

done


