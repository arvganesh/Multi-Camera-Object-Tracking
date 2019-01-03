#from __future__ import #print_function
from __future__ import division


from config import WORK_DIR, REID_PATH
import os
import sys
import time
import datetime
import os.path as osp
import numpy as np


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

sys.path.insert(0, (WORK_DIR + 'DEPENDENCIES/deeppersonreid'))

from torchreid.data_manager import ImageDataManager
from torchreid import models
from torchreid.losses import CrossEntropyLoss, DeepSupervision
from torchreid.utils.iotools import save_checkpoint, check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.loggers import Logger, RankLogger
from torchreid.utils.torchtools import count_num_param, open_all_layers, open_specified_layers
from torchreid.utils.reidtools import visualize_ranked_results
from torchreid.eval_metrics import evaluate
from torchreid.optimizers import init_optimizer


def person_reid():
    # model parameneters   
    use_gpus = False
    root = REID_PATH + "/data"
    name = "single_test"
    split_id = 0
    height = 256
    width = 128
    train_batch_size = 32
    test_batch_size = 1
    workers = 4
    train_sampler = ''
    num_instances = 4
    cuhk03_labeled = False
    cuhk03_classic_split = False

    torch.manual_seed(1)
    path_to_weights =  REID_PATH + "/weights/resnet50_market_xent/resnet50_market_xent.pth.tar"
    save_log_dir =  REID_PATH + "/log/eval-resnet50"

    log_name = 'log_test.txt'
    sys.stdout = Logger(osp.join(save_log_dir, log_name))

    # Initialize Data
    dm = ImageDataManager(use_gpus, [name], [name], root, split_id, height, width, train_batch_size, test_batch_size, workers, train_sampler, num_instances, cuhk03_labeled, cuhk03_classic_split)
    trainloader, testloader_dict = dm.return_dataloaders()

    # Initialize Model
    model = models.init_model("resnet50", num_classes=dm.num_train_pids, loss={'xent'}, use_gpu=use_gpus)

    # Initialize and Load Weights
    # from functools import partial
    # import pickle
    # pickle.load = partial(pickle.load, encoding="latin1")
    # pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    # checkpoint = torch.load(path_to_weights, pickle_module=pickle) 
    checkpoint = torch.load(path_to_weights, map_location='cpu')
    pretrain_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

    # Run Model
    queryloader = testloader_dict[name]['query']
    galleryloader = testloader_dict[name]['gallery']
    distmat = test(model, queryloader, galleryloader, use_gpus) 
    #visualize_ranked_results(distmat, dm.return_testdataset_by_name(name), save_log_dir + "/ranked_results/yeet", 1)
    #print ("DIST:", distmat)
    #gallery = dm.return_testdataset_by_name(name)[1]
    return distmat

def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 15, 20], return_distmat=True): # !!!!!!
    batch_time = AverageMeter()
    
    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)
            
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        # #print ("\n")
        # #print ("pid", q_pids)
        # #print ("camid", q_camids)
        # #print ("\n")
        gf, g_pids, g_camids = [], [], []
        end = time.time()
        for batch_idx, (imgs, pids, camids, _) in enumerate(galleryloader):

            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            # #print ("\n")
            # #print ("g_pid", pids)
            # #print ("g_camid", camids)

            features = model(imgs)
            # #print ("FEaT1", features)
            # #print ("FL1", len(features[0]))

            batch_time.update(time.time() - end)

            features = features.data.cpu()
            # #print ("FEaT", features)
            # #print ("FL", len(features[0]))
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        #print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # #print ("dist_b", distmat)

    distmat = distmat.addmm(beta=1, alpha=-2, mat1=qf, mat2=gf.t())

    distmat = distmat.numpy()
    # #print("dids: "+str(distmat))
    # #print ("W",distmat)
    indices = np.argsort(distmat, axis=1)
    # #print ("RESULT:", distmat[0][0], (distmat[0][0] < 0))
    # #print ("IND_E:", indices)
    # #print("Computing CMC and mAP")
    #match = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_cython=False)
    # #print("Results ----------")
    # #print("mAP: {:.1%}".format(mAP))
    # #print("CMC curve")
    # # for r in ranks:
    # #     from p#print import p#print
    # #     #print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    # # #print("------------------")
    
    #print ("match", match)
    if return_distmat:
        return distmat
    
    #print(cmc)
    #return ((distmat))
