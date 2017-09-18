#! /bin/bash

export SUBJECTS_DIR=

for time in  ffff
do
	for sub in vvv
	do
		
		set fmri = 
		set mprage = 
		set proc_anat = 
		
		if [-e $fmri] 
		then
			recon-all -s ${sub}-${time} -i ${mprage} -autorecon1 -parallel -openmp 4 -gcut
			
			mri_convert $SUBJECTS_DIR/${sub}-${time}/mri/brainmask.mgz $SUBJECTS_DIR/${sub}-${time}/mri/brain.nii
			set proc_anat = $SUBJECTS_DIR/${sub}-${time}/mri/brain.nii
			
			preprocessFunctional -4d ${fmri} -mprage_bet ${proc_anat} -bandpass_filter 0.01 0.1 \
			
			
		fi
	done
done
		