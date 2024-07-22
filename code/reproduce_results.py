from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.fastai_configs import *
#from configs.wavelet_configs import *
import numpy as np
import pandas as pd
import torch



    
datafolder = '../data/ptbxl/'
datafolder_icbeb = '../data/ICBEB/'
outputfolder = '../output/'

models = [conf_fastai_xresnet1d101_ST,
          #conf_decoupled_fastai_xresnet1d101,
          #conf_ST_fastai_inception1d,
          #conf_decoupled_fastai_inception1d,
          #conf_fastai_resnet1d_wang,
          #conf_decoupled_fastai_resnet1d_wang,
          #conf_fastai_lstm,
          #conf_decoupled_fastai_lstm,
          #conf_fastai_lstm_bidir,
          #conf_decoupled_fastai_lstm_bidir,
          #conf_fastai_fcn_wang
          ]

# ##########################################
# # STANDARD SCP EXPERIMENTS ON PTBXL
# ##########################################

# # experiments = [
# #     ('exp0', 'all'),
# #     ('exp1', 'diagnostic'),
# #     ('exp1.1', 'subdiagnostic'),
# #     ('exp1.1.1', 'superdiagnostic'),
# #     ('exp2', 'form'),
# #     ('exp3', 'rhythm')
# #    ]
experiments = [
    #('exp0', 'all'),
    ('exp1', 'diagnostic'),
    ('exp1.1', 'subdiagnostic'),
    ('exp1.1.1', 'superdiagnostic'),
    ]

# experiments = [
#     ('exp1.1.1', 'superdiagnostic')
#     ]
# experiments = [
#     ('exp1.1', 'subdiagnostic')
#     ]
# experiments = [
#     ('exp1', 'diagnostic')
#     ]

# # name = 'exp0'
# # task = 'all'





for name, task in experiments:
    e = SCP_Experiment(name, task, datafolder, outputfolder, models)
    e.prepare()
    e.perform()
    e.evaluate()
    modelnames = {
                  #'fastai_ST_xresnet1d101',
                  #'decoupled_fastai_xresnet1d101',
                  #'fastai_ST_inception1d',
                  #'decoupled_fastai_inception1d',
                  #'fastai_resnet1d_wang',
                  #'decoupled_fastai_resnet1d_wang',
                  #'fastai_lstm',
                  #'decoupled_fastai_lstm',
                  #'fastai_lstm_bidir',
                  #'decoupled_fastai_lstm_bidir',
                  'fastai_fcn_wang'
                  }
    utils.generate_ptbxl_summary_table_pri(models=modelnames)


# for name, task in experiments:
#     e = SCP_Experiment(name, task, datafolder, outputfolder, models)
#     e.prepare()
#     e.perform_wo_train()
#     e.evaluate()

# modelnames = {
#               'fastai_ST_xresnet1d101',
#               #'decoupled_fastai_xresnet1d101',
#               #'fastai_ST_inception1d',
#               #'decoupled_fastai_inception1d',
#               #'fastai_resnet1d_wang'
#               }
# utils.generate_ptbxl_summary_table_pri(models=modelnames)


name = 'exp1.1.1'
task = 'superdiagnostic'

models = [conf_decoupled_fastai_xresnet1d101]
e = SCP_Experiment(name, task, datafolder, outputfolder, models)
e.prepare()
pri = e.get_priMatrix(models[0])
pri = pri/torch.sum(pri,0)
pri = pri.numpy()
print(pri)
pri_df = pd.DataFrame(pri)
pri_df.to_excel('../'+task+'_pri.xlsx')

##########################################
# EXPERIMENT BASED ICBEB DATA
##########################################

# e = SCP_Experiment('exp_ICBEB', 'all', datafolder_icbeb, outputfolder, models)
# e.prepare()
# e.perform()
# e.evaluate()

# # generate greate summary table
# utils.ICBEBE_table()


    
