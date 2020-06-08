#!/bin/bash

ninety=90
seventytwo=72
two=2

for Subject in 21; do
#for Subject in 1 2 4 5 6 7 8 9 10 11 12 13 14 15 16 18 19 20 21 22 23; do

	if [ "$Subject" -gt "$two" ]; then
		slices=$seventytwo
	else
		slices=$ninety
	fi

    # Loop over directories
    for i in /home/andek67/Data/Gliom/Subject${Subject}/SCANS/*; do

	    echo "This is directory $i"	

	    file=`ls ${i}/DICOM/*.dcm | head -1`
	    description=`dicom_hdr ${file} | grep '0008 103e'`

    	if [[ $description == *"QMaps (T1T2PD)"* ]]; then

	    	echo "Found qMRI, lets check number of DICOM files"
       	 	nfiles=`ls ${i}/DICOM/*.dcm | wc -l`

        	if [ "$nfiles" -eq "$slices" ]; then
            	echo "Found the correct directory, dividing files into separate directories for T1, T2, PD"
            	mkdir ${i}/DICOM/T1 
            	mkdir ${i}/DICOM/T2 
            	mkdir ${i}/DICOM/PD

           		# Loop over DICOM files
            	for j in ${i}/DICOM/*.dcm; do
                	filetype=`dicom_hdr ${j} | grep '0008 0008'`
                	if [[ $filetype == *"QMAP\T1"* ]]; then
						cp $j ${i}/DICOM/T1
                	elif [[ $filetype == *"QMAP\T2"* ]]; then
                	    cp $j ${i}/DICOM/T2
                	else
                	    cp $j ${i}/DICOM/PD
                	fi
            	done

				# Make 3 nifti files for T1, T2, PD
            	# Check if this is qMRI with or without contrast
            	contrast=`dicom_hdr ${file} | grep '0018 0010'`
            	if [[ $contrast == *"ACQ Contrast"* ]]; then
					dcm2niix_afni -o /home/andek67/Data/Gliom/Subject${Subject} -f qMRI_T1_GD ${i}/DICOM/T1 
            		dcm2niix_afni -o /home/andek67/Data/Gliom/Subject${Subject} -f qMRI_T2_GD ${i}/DICOM/T2
            		dcm2niix_afni -o /home/andek67/Data/Gliom/Subject${Subject} -f qMRI_PD_GD ${i}/DICOM/PD
				else
					dcm2niix_afni -o /home/andek67/Data/Gliom/Subject${Subject} -f qMRI_T1 ${i}/DICOM/T1 
            		dcm2niix_afni -o /home/andek67/Data/Gliom/Subject${Subject} -f qMRI_T2 ${i}/DICOM/T2
            		dcm2niix_afni -o /home/andek67/Data/Gliom/Subject${Subject} -f qMRI_PD ${i}/DICOM/PD
				fi
        	else
		    	echo "This directory has $nfiles files instead of $slices"
			fi
    	elif [[ $description == *"NON"* ]]; then
			# Check if this is qMRI with or without contrast
            contrast=`dicom_hdr ${file} | grep '0018 0010'`
            if [[ $contrast == *"ACQ Contrast"* ]]; then
				dcm2niix_afni -o /home/andek67/Data/Gliom/Subject${Subject} -f NON_GD ${i}
			else
				dcm2niix_afni -o /home/andek67/Data/Gliom/Subject${Subject} -f NON ${i}
			fi
    	elif [[ $description == *"CSF"* ]]; then
			# Check if this is qMRI with or without contrast
            contrast=`dicom_hdr ${file} | grep '0018 0010'`
            if [[ $contrast == *"ACQ Contrast"* ]]; then
				dcm2niix_afni -o /home/andek67/Data/Gliom/Subject${Subject} -f CSF_GD ${i}
			else
				dcm2niix_afni -o /home/andek67/Data/Gliom/Subject${Subject} -f CSF ${i}
			fi
    	elif [[ $description == *"WM"* ]]; then
			# Check if this is qMRI with or without contrast
            contrast=`dicom_hdr ${file} | grep '0018 0010'`
            if [[ $contrast == *"ACQ Contrast"* ]]; then
				dcm2niix_afni -o /home/andek67/Data/Gliom/Subject${Subject} -f WM_GD ${i}
			else
				dcm2niix_afni -o /home/andek67/Data/Gliom/Subject${Subject} -f WM ${i}
			fi
    	elif [[ $description == *"GM"* ]]; then
			# Check if this is qMRI with or without contrast
            contrast=`dicom_hdr ${file} | grep '0018 0010'`
            if [[ $contrast == *"ACQ Contrast"* ]]; then
				dcm2niix_afni -o /home/andek67/Data/Gliom/Subject${Subject} -f GM_GD ${i}
			else
				dcm2niix_afni -o /home/andek67/Data/Gliom/Subject${Subject} -f GM ${i}
			fi
		fi

	done

done



