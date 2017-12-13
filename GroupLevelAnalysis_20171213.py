
# coding: utf-8

# In[ ]:

# LMEM for MRI data (3D nifti data)
def mri_lmem(model, mask, subject_dataframe, subject_files, grouping_variable):
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    from sys import stdout
    from os import getcwd
    
    import statsmodels.formula.api as smf
    from nibabel import load, save
    from numpy import array, empty_like, stack, nditer, zeros_like
    from pandas import DataFrame, read_csv, Series, concatenate
    
    working_dir = getcwd() + '/'
    subj_data = read_csv(subject_dataframe, header=0, index_col=0)

    # Load the brain data
    brain_niftis = [load(brain) for brain in subject_files]
    brain_data = [brain.get_data() for brain in brain_niftis]
    brain_data_4D = stack(brain_data, axis=3)
    
    # Load the mask
    mask_nifti = load(mask)
    mask = mask_nifti.get_data()
    
    ## Preallocate the output arrays
    # for the model
    ICCs_data = zeros_like(brain_data[0])
    pval_intercept_data = zeros_like(brain_data[0])
    pval_age_data = zeros_like(brain_data[0])
    pval_sex_data = zeros_like(brain_data[0])
    pval_ageSexInteract_data = zeros_like(brain_data[0])
    # per subject
    residuals_data = empty_like(brain_data_4D)
    fe_params_data = empty_like(brain_data_4D)
    re_params_data = empty_like(brain_data_4D)
    
    # Set up the actual loops
    coordinates = nditer(pval_intercept_data, None)
    for x,y,z,mod in coordinates:
        if mask[x][y][z] == 1:
            # Pull the subject brain data
            for a in range(0,len(brain_niftis)):
                brain[a] = brain_data[a][x][y][z]
            
            brain = Series(brain, index=subj_data.index, name='brain')
            data = concatenate([brain, subj_data],axis=1)
            # Run the mixed effects linear model
            mlm = smf.mixedlm(model, data, groups=data[grouping_variable])
            mod = mlm.fit()
            del [brain, data]
                
    
    # Save the ouputs as nifti files
    #output_data1 = [ICCs_data, pval_intercept_data, pval_invAge_data,
    #                pval_sex_data, pval_ageSexInteract_data]
    #output_data2 = [residuals_data, fe_params_data, re_params_data]
    #output_niftis1 = [Nifti1Image(result, brain_niftis[0].affine) for result in output_data1]
    ###### I don't think this one will work... test #######
    #output_niftis2 = [Nifti1Image(result, brain_niftis.affine) for result in output_data2]
    #output_niftis = output_niftis1 + output_niftis2
    
    #output_filenames = ['ICCs.nii','pval_intercept_data.nii','pval_age_data.nii',
    #                    'pval_sex_data.nii','pval_ageSexInteract_data.nii',
    #                    'residuals_data.nii','fe_params_data.nii','re_params_data.nii']
    #for a in output_niftis:
    #    save(a, working_dir + output_filenames(a.index))
    
    return(mod)


from glob import glob

analysis_home = '/Users/catcamacho/Box/LNCD_rewards_connectivity'
grouping_variable = 'timepoint'
model = 'brain ~ age + sex + age*sex'
subject_dataframe = analysis_home + '/misc/subjs_all.csv'
subject_files = glob(analysis_home + '/proc/firstlevel/smoothedMNI_conn_beta/*/%s/%s/')
mask = template_dir + '/MNI152_T1_3mm_brain.nii'

mod = mri_lmem(model, mask, subject_dataframe, subject_files, grouping_variable)
