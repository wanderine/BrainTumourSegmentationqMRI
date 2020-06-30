#!/bin/bash

for SubjectNumber in 1 2 4 5 6 7 8 9 10 11 12 13 14 15 16 18 19 20 21 22 23; do

	rm Subject${SubjectNumber}/brainmask_mask.nii.gz
	rm Subject${SubjectNumber}/brainmask.nii.gz
	rm Subject${SubjectNumber}/brainmask.nii

    #3dSkullStrip -input Subject${SubjectNumber}/T1FLAIR_GD.nii -prefix Subject${SubjectNumber}/brainmask_temp.nii -mask_vol -ld 50 -niter 750 -shrink_fac 0.85 &

    bet Subject${SubjectNumber}/T1FLAIR_GD.nii Subject${SubjectNumber}/brainmask.nii -m  -f 0.45 -R -B &

done

wait

for SubjectNumber in 1 2 4 5 6 7 8 9 10 11 12 13 14 15 16 18 19 20 21 22 23; do

    #3dcalc -a Subject${SubjectNumber}/brainmask_temp.nii -expr 'step(a)' -prefix Subject${SubjectNumber}/brainmask.nii
    rm Subject${SubjectNumber}/brainmask.nii.gz
    mv Subject${SubjectNumber}/brainmask_mask.nii.gz Subject${SubjectNumber}/brainmask.nii.gz

done
