
# coding: utf-8

# In[ ]:

# Import stuff
from os.path import join
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink, DataGrabber
from nipype.interfaces.fsl.utils import Merge, ImageMeants, Split
from nipype.interfaces.fsl.model import Randomise, Cluster
from nipype.interfaces.freesurfer.model import Binarize
from nipype.interfaces.fsl.maths import ApplyMask, Threshold
from pandas import DataFrame, Series

# MATLAB setup - Specify path to current SPM and the MATLAB's default mode
from nipype.interfaces.matlab import MatlabCommand
MatlabCommand.set_default_paths('~/spm12/toolbox')
MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")

# FSL set up- change default file output type
from nipype.interfaces.fsl import FSLCommand
FSLCommand.set_default_output_type('NIFTI')

# Set study variables
analysis_home = '/Users/catcamacho/Box/LNCD_rewards_connectivity'
secondLevel = analysis_home + '/proc/secondlevel'
out_dir = analysis_home + '/proc/secondlevel/final_models'
template_dir = analysis_home + '/templates'
template = template_dir + '/MNI152_T1_1mm_brain.nii'

#pull subject info 
subj_data = analysis_home + '/misc/subjs.csv'

conditions = ['punish','neutral']
seed_names = ['L_amyg','R_amyg']

# Group analysis models (predicting FC)
models = ['brain ~ ageMC + sex + ageMC*sex', 
          'brain ~ invAgeMC + sex + invAgeMC*sex']

model_names = ['linearAge', 'inverseAge']

terms = ['age', 'sex', 'ageSexInteract']



def finalize_models(clust_idx, out_dir, cluster, template, betas_text_file, subj_data, model_name):
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    from os.path import abspath
    import statsmodels.formula.api as smf
    from pandas import DataFrame, Series, concat, read_table, read_csv
    from ggplot import *
    from warnings import filterwarnings
    filterwarnings("ignore")
    import sys
    from nilearn import plotting

    origstdout = sys.stdout
    sys.stdout = open(out_dir+'/'+str(clust_idx)+'modelsummary.txt', 'w')

    #determine which model to use
    if model_name=='linearAge':
        model = 'brain ~ ageMC + sex + ageMC*sex'
    elif model_name=='inverseAge':
        model = 'brain ~ invAgeMC + sex + invAgeMC*sex'

    #organize data into dataframe for modeling
    subj_data = read_csv(subj_data, header=0, index_col=0)
    brain_data = read_table(betas_text_file, header=None, names=['brain'], index_col=None)
    alldata = concat([brain_data, subj_data],axis=1)
    # do the modeling
    mlm = smf.mixedlm(model, alldata, groups=alldata['Timepoint'])
    mod = mlm.fit()
    print(mod.summary())

    sys.stdout = origstdout
    summary_file = out_dir+'/'+str(clust_idx)+'modelsummary.txt'

    # plot the model results
    figure = ggplot(alldata, aes(x='age',y='brain')) 
    figure = figure + geom_point()
    figure.save(out_dir+'/'+str(clust_idx)+'plot.svg')
    figure_file = out_dir+'/'+str(clust_idx)+'plot.svg'

    # make a picture of the brain cluster
    display = plotting.plot_anat(anat_img = template, display_mode='x')
    display.add_overlay(cluster)
    display.savefig(out_dir+'/'+str(clust_idx)+'clusterpic.png')
    display.close()
    clusterpic_file = out_dir+'/'+str(clust_idx)+'clusterpic.png'

    outputs = [summary_file,figure_file, clusterpic_file]

    return(outputs)

from glob import glob
from os import makedirs
from os.path import exists

for a in conditions:
    for b in seed_names:
        for c in model_names:
            for d in terms:
                currout_dir = out_dir + '/%s_%s_%s_%s' % (a,b,c,d)
                if not exists(currout_dir):
                    makedirs(currout_dir)
                clusters = glob(secondLevel + '/final_clusters/%s_model_%s%s_term_%s/*' % (a,c,b,d))
                for i in range(0,len(clusters)):
                    betas_text_file = glob(secondLevel + '/beta_values/%s_model_%s%s_term_%s/_pull_mean_betas%d/mean_connectivity.txt' % (a,c,b,d,i))
                    cluster = clusters[i]
                    finalize_models(i, currout_dir, cluster, template, betas_text_file[0], subj_data, c)     




