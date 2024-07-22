from models.timeseries_utils import *

from fastai import *
from fastai.basic_data import *
from fastai.basic_train import *
from fastai.train import *
from fastai.metrics import *
from fastai.torch_core import *
from fastai.callbacks.tracker import SaveModelCallback
from torch.nn.functional import *

from pathlib import Path
from functools import partial

from models.resnet1d import resnet1d18,resnet1d34,resnet1d50,resnet1d101,resnet1d152,resnet1d_wang,resnet1d,wrn1d_22,resnet1d_wang_decoupled
from models.xresnet1d import xresnet1d18,xresnet1d34,xresnet1d50,xresnet1d101,xresnet1d152,xresnet1d18_deep,xresnet1d34_deep,xresnet1d50_deep,xresnet1d18_deeper,xresnet1d34_deeper,xresnet1d50_deeper
from models.inception1d import inception1d_decoupled
from models.basic_conv1d import fcn,fcn_wang,schirrmeister,sen,basic1d,weight_init
from models.rnn1d import RNN1d,RNN1d_decoupled
from models.aver_xresnet1d import xresnet1d101_aver
from models.L_aver_xresnet1d import xresnet1d101_aver_L
from models.mask_xresnet1d import xresnet1d101_mask
from models.camask_xresnet1d import xresnet1d101_maskca
from models.ST_xresnet1d import xresnet1d101_ST
from models.ST_inception1d import inception1d_ST
import math

from models.basic_conv1d_decoupled import fcn,fcn_wang,schirrmeister,sen,basic1d,weight_init, fcn_wang_decoupled

from models.xresnet1d_decoupled import xresnet1d18,xresnet1d34,xresnet1d50,xresnet1d101,xresnet1d152,xresnet1d18_deep,xresnet1d34_deep,xresnet1d50_deep,xresnet1d18_deeper,xresnet1d34_deeper,xresnet1d50_deeper, xresnet1d101_decoupled

from models.base_model import ClassificationModel
import torch
import torch.nn as nn

#for lrfind
import matplotlib
import matplotlib.pyplot as plt

#eval for early stopping
from fastai.callback import Callback
from utils.utils import evaluate_experiment

class focal_loss(nn.Module):
    def __init__(self,gamma=2,alpha=0.80):
        super(focal_loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self,logits,targets):
        # ll = logits.reshape(-1)
        # tt = targets.reshape(-1)
        # pp = torch.sigmoid(ll)
        # pp = torch.where(tt>=0.5,pp,1-pp)
        # logp = -torch.log(pp)
        # loss = logp*((1-pp)**self.gamma)
        # loss = loss.mean()
        # loss.mean()
        
        logits_sigmoid = logits.sigmoid()
        pt = (1-logits_sigmoid)*targets+logits_sigmoid*(1-targets) # actually it is 1-pt in the paper
        
        #focal_weight = (self.alpha*targets+(1-self.alpha)*(1-targets))*pt.pow(self.gamma)
        focal_weight = pt**self.gamma
        #losstmp = nn.functional.binary_cross_entropy_with_logits(logits, targets,reduction='none')
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets,reduction='none')*focal_weight
        #lossm,_ = torch.max(loss,dim=1,keepdim=True)
        return loss.mean()
    
class focal_loss_soft(nn.Module):
    def __init__(self,gamma=2,alpha=0.80):
        super(focal_loss_soft, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.counter = 0
        self.newEpch = False
        self.iter = 0
    def forward(self,logits,targets):

        # self.iter = self.iter + 1
        logits_sigmoid = logits.sigmoid()
        pt = (1-logits_sigmoid)*targets+logits_sigmoid*(1-targets) # actually it is 1-pt in the paper
        
        focal_weight = pt**self.gamma
        
        # if self.counter > 50:
        #     x=2
        
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets,reduction='none')*focal_weight
        
        dropout = (torch.rand(1,23)>0.005).float()
        
        loss = loss*dropout.cuda()
        
        
        # if self.counter >50:
        #     thre = 0.80
        #     loss[pt>=thre] = loss[pt>=thre]*0.5
        
        # if targets.shape[0] != 128:
        #     self.counter = self.counter + 1
        #     self.newEpch = True
        #     self.iter = 0
        return loss.mean()


class metric_func(Callback):
    "Obtains score using user-supplied function func (potentially ignoring targets with ignore_idx)"
    def __init__(self, func, name="metric_func", ignore_idx=None, one_hot_encode_target=True, argmax_pred=False, softmax_pred=True, flatten_target=True, sigmoid_pred=False,metric_component=None):
        super().__init__()
        self.func = func
        self.ignore_idx = ignore_idx
        self.one_hot_encode_target = one_hot_encode_target
        self.argmax_pred = argmax_pred
        self.softmax_pred = softmax_pred
        self.flatten_target = flatten_target
        self.sigmoid_pred = sigmoid_pred
        self.metric_component = metric_component
        self.name=name

    def on_epoch_begin(self, **kwargs):
        self.y_pred = None
        self.y_true = None
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        #flatten everything (to make it also work for annotation tasks)
        y_pred_flat = last_output.view((-1,last_output.size()[-1]))
        
        if(self.flatten_target):
            y_true_flat = last_target.view(-1)
        y_true_flat = last_target

        #optionally take argmax of predictions
        if(self.argmax_pred is True):
            y_pred_flat = y_pred_flat.argmax(dim=1)
        elif(self.softmax_pred is True):
            y_pred_flat = F.softmax(y_pred_flat, dim=1)
        elif(self.sigmoid_pred is True):
            y_pred_flat = torch.sigmoid(y_pred_flat)
        
        #potentially remove ignore_idx entries
        if(self.ignore_idx is not None):
            selected_indices = (y_true_flat!=self.ignore_idx).nonzero().squeeze()
            y_pred_flat = y_pred_flat[selected_indices]
            y_true_flat = y_true_flat[selected_indices]
        
        y_pred_flat = to_np(y_pred_flat)
        y_true_flat = to_np(y_true_flat)

        if(self.one_hot_encode_target is True):
            y_true_flat = one_hot_np(y_true_flat,last_output.size()[-1])

        if(self.y_pred is None):
            self.y_pred = y_pred_flat
            self.y_true = y_true_flat
        else:
            self.y_pred = np.concatenate([self.y_pred, y_pred_flat], axis=0)
            self.y_true = np.concatenate([self.y_true, y_true_flat], axis=0)
    
    def on_epoch_end(self, last_metrics, **kwargs):
        #access full metric (possibly multiple components) via self.metric_complete
        self.metric_complete = self.func(self.y_true, self.y_pred)
        if(self.metric_component is not None):
            return add_metrics(last_metrics, self.metric_complete[self.metric_component])
        else:
            return add_metrics(last_metrics, self.metric_complete)

def fmax_metric(targs,preds):
    return evaluate_experiment(targs,preds)["Fmax"]

def auc_metric(targs,preds):
    return evaluate_experiment(targs,preds)["macro_auc"]

def mse_flat(preds,targs):
    return torch.mean(torch.pow(preds.view(-1)-targs.view(-1),2))

def nll_regression(preds,targs):
    #preds: bs, 2
    #targs: bs, 1
    preds_mean = preds[:,0]
    #warning: output goes through exponential map to ensure positivity
    preds_var = torch.clamp(torch.exp(preds[:,1]),1e-4,1e10)
    #print(to_np(preds_mean)[0],to_np(targs)[0,0],to_np(torch.sqrt(preds_var))[0])
    return torch.mean(torch.log(2*math.pi*preds_var)/2) + torch.mean(torch.pow(preds_mean-targs[:,0],2)/2/preds_var)
    
def nll_regression_init(m):
    assert(isinstance(m, nn.Linear))
    nn.init.normal_(m.weight,0.,0.001)
    nn.init.constant_(m.bias,4)

def lr_find_plot(learner, path, filename="lr_find", n_skip=10, n_skip_end=2):
    '''saves lr_find plot as file (normally only jupyter output)
    on the x-axis is lrs[-1]
    '''
    learner.lr_find()
    
    backend_old= matplotlib.get_backend()
    plt.switch_backend('agg')
    plt.ylabel("loss")
    plt.xlabel("learning rate (log scale)")
    losses = [ to_np(x) for x in learner.recorder.losses[n_skip:-(n_skip_end+1)]]
    #print(learner.recorder.val_losses)
    #val_losses = [ to_np(x) for x in learner.recorder.val_losses[n_skip:-(n_skip_end+1)]]

    plt.plot(learner.recorder.lrs[n_skip:-(n_skip_end+1)],losses )
    #plt.plot(learner.recorder.lrs[n_skip:-(n_skip_end+1)],val_losses )

    plt.xscale('log')
    plt.savefig(str(path/(filename+'.png')))
    plt.switch_backend(backend_old)

def losses_plot(learner, path, filename="losses", last:int=None):
    '''saves lr_find plot as file (normally only jupyter output)
    on the x-axis is lrs[-1]
    '''
    backend_old= matplotlib.get_backend()
    plt.switch_backend('agg')
    plt.ylabel("loss")
    plt.xlabel("Batches processed")

    last = ifnone(last,len(learner.recorder.nb_batches))
    l_b = np.sum(learner.recorder.nb_batches[-last:])
    iterations = range_of(learner.recorder.losses)[-l_b:]
    plt.plot(iterations, learner.recorder.losses[-l_b:], label='Train')
    val_iter = learner.recorder.nb_batches[-last:]
    val_iter = np.cumsum(val_iter)+np.sum(learner.recorder.nb_batches[:-last])
    plt.plot(val_iter, learner.recorder.val_losses[-last:], label='Validation')
    plt.legend()

    plt.savefig(str(path/(filename+'.png')))
    plt.switch_backend(backend_old)
    
def pairwise_distance_torch(embeddings):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      embeddings: 2-D Tensor of size [number of data, feature dimension].
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """

    # pairwise distance matrix with precise embeddings
    precise_embeddings = embeddings.to(dtype=torch.float32)

    c1 = torch.pow(precise_embeddings, 2).sum(axis=-1)
    c2 = torch.pow(precise_embeddings.transpose(0, 1), 2).sum(axis=0)
    c3 = precise_embeddings @ precise_embeddings.transpose(0, 1)

    c1 = c1.reshape((c1.shape[0], 1))
    c2 = c2.reshape((1, c2.shape[0]))
    c12 = c1 + c2
    pairwise_distances_squared = c12 - 2.0 * c3

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.max(pairwise_distances_squared, torch.tensor([0.]).cuda())
    # Get the mask where the zero distances are at.
    error_mask = pairwise_distances_squared.clone()
    error_mask[error_mask > 0.0] = 1.
    error_mask[error_mask <= 0.0] = 0.

    pairwise_distances = torch.mul(pairwise_distances_squared, error_mask)

    # Explicitly set diagonals to zero.
    mask_offdiagonals = torch.ones((pairwise_distances.shape[0], pairwise_distances.shape[1])) - torch.diag(
        torch.ones(pairwise_distances.shape[0]))
    pairwise_distances = torch.mul(pairwise_distances.cuda(), mask_offdiagonals.cuda())
    return pairwise_distances


def TripletSemiHardLoss(y_true, y_pred, margin=1.0):
    """Computes the triplet loss_functions with semi-hard negative mining.
       The loss_functions encourages the positive distances (between a pair of embeddings
       with the same labels) to be smaller than the minimum negative distance
       among which are at least greater than the positive distance plus the
       margin constant (called semi-hard negative) in the mini-batch.
       If no such negative exists, uses the largest negative distance instead.
       See: https://arxiv.org/abs/1503.03832.
       We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
       [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
       2-D float `Tensor` of l2 normalized embedding vectors.
       Args:
         margin: Float, margin term in the loss_functions definition. Default value is 1.0.
         name: Optional name for the op.
       """

    labels, embeddings = y_true, y_pred

    # Reshape label tensor to [batch_size, 1].
    lshape = labels.shape
    labels = torch.reshape(labels, [lshape[0], 1])

    pdist_matrix = pairwise_distance_torch(embeddings)

    # Build pairwise binary adjacency matrix.
    adjacency = torch.eq(labels, labels.transpose(0, 1))
    # Invert so we can select negatives only.
    adjacency_not = adjacency.logical_not()

    batch_size = labels.shape[0]

    # Compute the mask.
    pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
    adjacency_not_tile = adjacency_not.repeat(batch_size, 1)

    transpose_reshape = pdist_matrix.transpose(0, 1).reshape(-1, 1)
    greater = pdist_matrix_tile > transpose_reshape

    mask = adjacency_not_tile & greater

    # final mask
    mask_step = mask.to(dtype=torch.float32)
    mask_step = mask_step.sum(axis=1)
    mask_step = mask_step > 0.0
    mask_final = mask_step.reshape(batch_size, batch_size)
    mask_final = mask_final.transpose(0, 1)

    adjacency_not = adjacency_not.to(dtype=torch.float32)
    mask = mask.to(dtype=torch.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    axis_maximums = torch.max(pdist_matrix_tile, dim=1, keepdim=True)
    masked_minimums = torch.min(torch.mul(pdist_matrix_tile - axis_maximums[0], mask), dim=1, keepdim=True)[0] + \
                      axis_maximums[0]
    negatives_outside = masked_minimums.reshape([batch_size, batch_size])
    negatives_outside = negatives_outside.transpose(0, 1)

    # negatives_inside: largest D_an.
    axis_minimums = torch.min(pdist_matrix, dim=1, keepdim=True)
    masked_maximums = torch.max(torch.mul(pdist_matrix - axis_minimums[0], adjacency_not), dim=1, keepdim=True)[0] + \
                      axis_minimums[0]
    negatives_inside = masked_maximums.repeat(1, batch_size)

    semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = margin + pdist_matrix - semi_hard_negatives

    mask_positives = adjacency.to(dtype=torch.float32) - torch.diag(torch.ones(batch_size)).cuda()
    num_positives = mask_positives.sum()

    triplet_loss = (torch.max(torch.mul(loss_mat, mask_positives), torch.tensor([0.]).cuda())).sum() / num_positives
    triplet_loss = triplet_loss.to(dtype=embeddings.dtype)
    return triplet_loss


def multi_triplet_loss(alpha=1., beta=0.0, gamma=0.01, margin=0.5):
    def loss(multi, y_true):

        logits, metric_logit = multi

        y_pred = sigmoid(logits)

        y_true_1 = y_true
        y_pred_1 = y_pred

        y_true_2 = y_true
        y_pred_2 = y_pred

        y_pred_1 = torch.clamp(y_pred_1, 1e-7, 1.0 - 1e-7)
        y_true_2 = torch.clamp(y_true_2, 1e-4, 1.0 - 1e-4)
        
        pt = (1-y_pred)*y_true+y_pred*(1-y_true) # actually it is 1-pt in the paper
        
        #focal_weight = pt**2

        focal_weight = 1

        cls = F.binary_cross_entropy_with_logits(input=logits, target=y_true,reduction='none')*focal_weight
        rls = torch.mean(- y_pred_2 * torch.log(y_true_2) - (1 - y_pred_2) * torch.log(1 - y_true_2))

        mls = 0

        for num in range(y_true.shape[-1]):
            sub_y_true = y_true[:,num]
            mls += TripletSemiHardLoss(y_true=sub_y_true, y_pred=metric_logit[num], margin=margin)
        mls = mls / y_true.shape[-1]

        return alpha * cls.mean() + beta * rls + gamma * mls

    return loss

class fastai_model(ClassificationModel):
    def __init__(self,name,n_classes,freq,outputfolder,input_shape,pretrained=False,input_size=2.5,input_channels=12,chunkify_train=False,chunkify_valid=True,bs=128,ps_head=0.5,lin_ftrs_head=[128],div_lin_ftrs_head=[16],wd=1e-2,epochs=50,lr=1e-2,kernel_size=5,loss="binary_cross_entropy",pretrainedfolder=None,n_classes_pretrained=None,gradual_unfreezing=True,discriminative_lrs=True,epochs_finetuning=30,early_stopping=None,aggregate_fn="max",concat_train_val=False, alpha=1., beta=0.0, gamma=0.01, margin=0.5):
        super().__init__()
        
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.margin=margin
        
        self.name = name
        self.num_classes = n_classes if loss!= "nll_regression" else 2
        self.target_fs = freq
        self.outputfolder = Path(outputfolder)

        self.input_size=int(input_size*self.target_fs)
        self.input_channels=input_channels

        self.chunkify_train=chunkify_train
        self.chunkify_valid=chunkify_valid

        self.chunk_length_train=2*self.input_size#target_fs*6
        self.chunk_length_valid=self.input_size

        self.min_chunk_length=self.input_size#chunk_length

        self.stride_length_train=self.input_size#chunk_length_train//8
        self.stride_length_valid=self.input_size//2#chunk_length_valid

        self.copies_valid = 0 #>0 should only be used with chunkify_valid=False
        
        self.bs=bs
        self.ps_head=ps_head
        self.lin_ftrs_head=lin_ftrs_head
        self.div_lin_ftrs_head = div_lin_ftrs_head
        self.wd=wd
        self.epochs=epochs
        self.lr=lr
        self.kernel_size = kernel_size
        self.loss = loss
        self.input_shape = input_shape

        if pretrained == True:
            if(pretrainedfolder is None):
                pretrainedfolder = Path('../output/exp0/models/'+name.split("_pretrained")[0]+'/')
            if(n_classes_pretrained is None):
                n_classes_pretrained = 71
  
        self.pretrainedfolder = None if pretrainedfolder is None else Path(pretrainedfolder)
        self.n_classes_pretrained = n_classes_pretrained
        self.discriminative_lrs = discriminative_lrs
        self.gradual_unfreezing = gradual_unfreezing
        self.epochs_finetuning = epochs_finetuning

        self.early_stopping = early_stopping
        self.aggregate_fn = aggregate_fn
        self.concat_train_val = concat_train_val

    def fit(self, X_train, y_train, X_val, y_val):
        #convert everything to float32
        X_train = [l.astype(np.float32) for l in X_train]
        X_val = [l.astype(np.float32) for l in X_val]
        y_train = [l.astype(np.float32) for l in y_train]
        y_val = [l.astype(np.float32) for l in y_val]

        if(self.concat_train_val):
            X_train += X_val
            y_train += y_val
        
        if(self.pretrainedfolder is None): #from scratch
            print("Training from scratch...")
            learn = self._get_learner(X_train,y_train,X_val,y_val,if_train=True)
            
            #if(self.discriminative_lrs):
            #    layer_groups=learn.model.get_layer_groups()
            #    learn.split(layer_groups)
            learn.model.apply(weight_init)
            
            #initialization for regression output
            if(self.loss=="nll_regression" or self.loss=="mse"):
                output_layer_new = learn.model.get_output_layer()
                output_layer_new.apply(nll_regression_init)
                learn.model.set_output_layer(output_layer_new)
            
            lr_find_plot(learn, self.outputfolder)
            learn.fit_one_cycle(self.epochs,self.lr)#slice(self.lr) if self.discriminative_lrs else self.lr)

            # 使用savemodelcallback最后会加载bestmodel
            # learn.fit_one_cycle(self.epochs,self.lr,callbacks=SaveModelCallback(learn,monitor='metric_func',name='optimal'))#slice(self.lr) if self.discriminative_lrs else self.lr)
            losses_plot(learn, self.outputfolder)
        else: #finetuning
            print("Finetuning...")
            #create learner
            learn = self._get_learner(X_train,y_train,X_val,y_val,self.n_classes_pretrained)
            
            #load pretrained model
            learn.path = self.pretrainedfolder
            learn.load(self.pretrainedfolder.stem)
            learn.path = self.outputfolder

            #exchange top layer
            output_layer = learn.model.get_output_layer()
            output_layer_new = nn.Linear(output_layer.in_features,self.num_classes).cuda()
            apply_init(output_layer_new, nn.init.kaiming_normal_)
            learn.model.set_output_layer(output_layer_new)
            
            #layer groups
            if(self.discriminative_lrs):
                layer_groups=learn.model.get_layer_groups()
                learn.split(layer_groups)

            learn.train_bn = True #make sure if bn mode is train
            
            
            #train
            lr = self.lr
            if(self.gradual_unfreezing):
                assert(self.discriminative_lrs is True)
                learn.freeze()
                lr_find_plot(learn, self.outputfolder,"lr_find0")
                learn.fit_one_cycle(self.epochs_finetuning,lr)
                losses_plot(learn, self.outputfolder,"losses0")
                #for n in [0]:#range(len(layer_groups)):
                #    learn.freeze_to(-n-1)
                #    lr_find_plot(learn, self.outputfolder,"lr_find"+str(n))
                #    learn.fit_one_cycle(self.epochs_gradual_unfreezing,slice(lr))
                #    losses_plot(learn, self.outputfolder,"losses"+str(n))
                    #if(n==0):#reduce lr after first step
                    #    lr/=10.
                    #if(n>0 and (self.name.startswith("fastai_lstm") or self.name.startswith("fastai_gru"))):#reduce lr further for RNNs
                    #    lr/=10
                    
            learn.unfreeze()
            lr_find_plot(learn, self.outputfolder,"lr_find"+str(len(layer_groups)))
            learn.fit_one_cycle(self.epochs_finetuning,slice(lr/1000,lr/10))
            losses_plot(learn, self.outputfolder,"losses"+str(len(layer_groups)))

        learn.save(self.name) #even for early stopping the best model will have been loaded again

    def predict(self, X):
        X = [l.astype(np.float32) for l in X]
        y_dummy = [np.ones(self.num_classes,dtype=np.float32) for _ in range(len(X))]
        
        learn = self._get_learner(X,y_dummy,X,y_dummy,if_train=False)
        learn.load(self.name)
        
        preds,targs=learn.get_preds()
        preds=to_np(preds)
        
        idmap=learn.data.valid_ds.get_id_mapping()

        return aggregate_predictions(preds,idmap=idmap,aggregate_fn = np.mean if self.aggregate_fn=="mean" else np.amax)  

    def predict_opt(self, X):
        X = [l.astype(np.float32) for l in X]
        y_dummy = [np.ones(self.num_classes,dtype=np.float32) for _ in range(len(X))]
        
        learn = self._get_learner(X,y_dummy,X,y_dummy)
        learn.load('optimal')
        
        preds,targs=learn.get_preds()
        preds=to_np(preds)
        
        idmap=learn.data.valid_ds.get_id_mapping()

        return aggregate_predictions(preds,idmap=idmap,aggregate_fn = np.mean if self.aggregate_fn=="mean" else np.amax)  
 
        
    def _get_learner(self, X_train,y_train,X_val,y_val,if_train,num_classes=None):
        df_train = pd.DataFrame({"data":range(len(X_train)),"label":y_train})
        df_valid = pd.DataFrame({"data":range(len(X_val)),"label":y_val})
        
        tfms_ptb_xl = [ToTensor()]
                
        ds_train=TimeseriesDatasetCrops(df_train,self.input_size,num_classes=self.num_classes,chunk_length=self.chunk_length_train if self.chunkify_train else 0,min_chunk_length=self.min_chunk_length,stride=self.stride_length_train,transforms=tfms_ptb_xl,annotation=False,col_lbl ="label",npy_data=X_train)
        ds_valid=TimeseriesDatasetCrops(df_valid,self.input_size,num_classes=self.num_classes,chunk_length=self.chunk_length_valid if self.chunkify_valid else 0,min_chunk_length=self.min_chunk_length,stride=self.stride_length_valid,transforms=tfms_ptb_xl,annotation=False,col_lbl ="label",npy_data=X_val)
    
        db = DataBunch.create(ds_train,ds_valid,bs=self.bs)

        if (self.loss == "binary_cross_entropy"):
            loss = F.binary_cross_entropy_with_logits
        elif (self.loss == "decoupled_triplet"):
            loss = multi_triplet_loss(alpha=self.alpha, beta=self.beta, gamma=self.gamma, margin=self.margin)
        elif (self.loss == "cross_entropy"):
            loss = F.cross_entropy
        elif (self.loss == "mse"):
            loss = mse_flat
        elif (self.loss == "nll_regression"):
            loss = nll_regression
        else:
            print("loss not found")
            assert (True)
               
        self.input_channels = self.input_shape[-1]
        metrics = []

        print("model:",self.name) #note: all models of a particular kind share the same prefix but potentially a different postfix such as _input256
        num_classes = self.num_classes if num_classes is None else num_classes
        #resnet resnet1d18,resnet1d34,resnet1d50,resnet1d101,resnet1d152,resnet1d_wang,resnet1d,wrn1d_22
        if(self.name.startswith("fastai_resnet1d18")):
            model = resnet1d18(num_classes=num_classes,input_channels=self.input_channels,inplanes=128,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_resnet1d34")):
            model = resnet1d34(num_classes=num_classes,input_channels=self.input_channels,inplanes=128,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_resnet1d50")):
            model = resnet1d50(num_classes=num_classes,input_channels=self.input_channels,inplanes=128,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_resnet1d101")):
            model = resnet1d101(num_classes=num_classes,input_channels=self.input_channels,inplanes=128,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_resnet1d152")):
            model = resnet1d152(num_classes=num_classes,input_channels=self.input_channels,inplanes=128,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_resnet1d_wang")):
            model = resnet1d_wang(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_wrn1d_22")):    
            model = wrn1d_22(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif (self.name.startswith("decoupled_fastai_resnet1d_wang")):
            model = resnet1d_wang_decoupled(num_classes=num_classes, input_channels=self.input_channels,
                                  kernel_size=self.kernel_size, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head, div_lin_ftrs_head=self.div_lin_ftrs_head, if_train=if_train)
        
            
        #xresnet ... (order important for string capture)
        elif(self.name.startswith("fastai_xresnet1d18_deeper")):
            model = xresnet1d18_deeper(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_xresnet1d34_deeper")):
            model = xresnet1d34_deeper(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_xresnet1d50_deeper")):
            model = xresnet1d50_deeper(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_xresnet1d18_deep")):
            model = xresnet1d18_deep(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_xresnet1d34_deep")):
            model = xresnet1d34_deep(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_xresnet1d50_deep")):
            model = xresnet1d50_deep(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_xresnet1d18")):
            model = xresnet1d18(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_xresnet1d34")):
            model = xresnet1d34(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_xresnet1d50")):
            model = xresnet1d50(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_xresnet1d101")):
            model = xresnet1d101(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_xresnet1d152")):
            model = xresnet1d152(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_aver_xresnet1d101")):
            model = xresnet1d101_aver(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_L_aver_xresnet1d101")):
            model = xresnet1d101_aver_L(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_mask_xresnet1d101")):
            model = xresnet1d101_mask(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_CA_mask_xresnet1d101")):
            model = xresnet1d101_maskca(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_ST_xresnet1d101")):
            model = xresnet1d101_ST(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif (self.name.startswith("decoupled_fastai_xresnet1d101")):
            model = xresnet1d101_decoupled(num_classes=num_classes, input_channels=self.input_channels,
                                 kernel_size=self.kernel_size, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head, div_lin_ftrs_head=self.div_lin_ftrs_head, if_train=if_train)

               
        #inception
        #passing the default kernel size of 5 leads to a max kernel size of 40-1 in the inception model as proposed in the original paper
        elif(self.name == "fastai_inception1d_no_residual"):#note: order important for string capture
            model = inception1d(num_classes=num_classes,input_channels=self.input_channels,use_residual=False,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head,kernel_size=8*self.kernel_size)
        elif(self.name.startswith("fastai_inception1d")):
            model = inception1d(num_classes=num_classes,input_channels=self.input_channels,use_residual=True,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head,kernel_size=8*self.kernel_size)
        elif(self.name.startswith("fastai_ST_inception1d")):
            model = inception1d_ST(num_classes=num_classes,input_channels=self.input_channels,use_residual=True,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head,kernel_size=8*self.kernel_size)
        elif (self.name.startswith("decoupled_fastai_inception1d")):
            model = inception1d_decoupled(num_classes=num_classes, input_channels=self.input_channels, use_residual=True,
                                ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head, div_lin_ftrs_head=self.div_lin_ftrs_head,
                                kernel_size=8 * self.kernel_size, if_train=if_train)


        #basic_conv1d fcn,fcn_wang,schirrmeister,sen,basic1d
        elif(self.name.startswith("fastai_fcn_wang")):#note: order important for string capture
            model = fcn_wang(num_classes=num_classes,input_channels=self.input_channels,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_fcn")):
            model = fcn(num_classes=num_classes,input_channels=self.input_channels)
        elif(self.name.startswith("fastai_schirrmeister")):
            model = schirrmeister(num_classes=num_classes,input_channels=self.input_channels,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_sen")):
            model = sen(num_classes=num_classes,input_channels=self.input_channels,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_basic1d")):    
            model = basic1d(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        #RNN
        elif(self.name.startswith("fastai_lstm_bidir")):
            model = RNN1d(input_channels=self.input_channels,num_classes=num_classes,lstm=True,bidirectional=True,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_gru_bidir")):
            model = RNN1d(input_channels=self.input_channels,num_classes=num_classes,lstm=False,bidirectional=True,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_lstm")):
            model = RNN1d(input_channels=self.input_channels,num_classes=num_classes,lstm=True,bidirectional=False,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_gru")):
            model = RNN1d(input_channels=self.input_channels,num_classes=num_classes,lstm=False,bidirectional=False,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif (self.name.startswith("decoupled_fastai_lstm")):
            model = RNN1d_decoupled(input_channels=self.input_channels, num_classes=num_classes, lstm=True, bidirectional=False,
                          ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head, div_lin_ftrs_head=self.div_lin_ftrs_head, if_train=if_train)
        elif (self.name.startswith("decoupled_fastai_lstm_bidir")):
            model = RNN1d_decoupled(input_channels=self.input_channels, num_classes=num_classes, lstm=True, bidirectional=True,
                          ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head, div_lin_ftrs_head=self.div_lin_ftrs_head, if_train=if_train)
     
        else:
            print("Model not found.")
            assert(True)
        
        # count=0
        
        # for p in model.named_parameters():
        #     if count==0:
        #         print(p)
        #         break
        #     count = count+1
        
        #learn = Learner(db,model, loss_func=self.loss, metrics=metrics,wd=self.wd,path=self.outputfolder)
        #loss = multi_triplet_loss(alpha=1, beta=0, gamma=0.1, margin=2)
        metric = metric_func(auc_metric, one_hot_encode_target=False, argmax_pred=False, softmax_pred=False, sigmoid_pred=True, flatten_target=False)
        #learn = Learner(db,model, loss_func=loss, metrics=[],wd=self.wd,path=self.outputfolder)
        learn = Learner(db,model, loss_func=focal_loss(), metrics=[],wd=self.wd,path=self.outputfolder)
        
        #learn.callback_fns.append(partial(SaveModelCallback, monitor="macro_auc", name="optimal"))
        
        if(self.name.startswith("fastai_lstm") or self.name.startswith("fastai_gru")):
            learn.callback_fns.append(partial(GradientClipping, clip=0.25))

        if(self.early_stopping is not None):
            #supported options: valid_loss, macro_auc, fmax
            if(self.early_stopping == "macro_auc" and self.loss != "mse" and self.loss !="nll_regression"):
                metric = metric_func(auc_metric, self.early_stopping, one_hot_encode_target=False, argmax_pred=False, softmax_pred=False, sigmoid_pred=True, flatten_target=False)
                learn.metrics.append(metric)
                learn.callback_fns.append(partial(SaveModelCallback, monitor=self.early_stopping, every='improvement', name=self.name))
            elif(self.early_stopping == "fmax" and self.loss != "mse" and self.loss !="nll_regression"):
                metric = metric_func(fmax_metric, self.early_stopping, one_hot_encode_target=False, argmax_pred=False, softmax_pred=False, sigmoid_pred=True, flatten_target=False)
                learn.metrics.append(metric)
                learn.callback_fns.append(partial(SaveModelCallback, monitor=self.early_stopping, every='improvement', name=self.name))
            elif(self.early_stopping == "valid_loss"):
                learn.callback_fns.append(partial(SaveModelCallback, monitor=self.early_stopping, every='improvement', name=self.name))
            
        return learn
