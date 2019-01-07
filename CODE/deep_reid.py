#from __future__ import #print_function
from __future__ import division

from config import *
# from PIL import Image
# import torch
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# from torch.optim import lr_scheduler

# sys.path.insert(0, (WORK_DIR + 'DEPENDENCIES/deeppersonreid'))

# from torchreid.data_manager import ImageDataManager
# from torchreid import models
# from torchreid.losses import CrossEntropyLoss, DeepSupervision
# from torchreid.utils.iotools import save_checkpoint, check_isfile
# from torchreid.utils.avgmeter import AverageMeter
# from torchreid.utils.loggers import Logger, RankLogger
# from torchreid.utils.torchtools import count_num_param, open_all_layers, open_specified_layers
# from torchreid.utils.reidtools import visualize_ranked_results
# from torchreid.eval_metrics import evaluate
# from torchreid.optimizers import init_optimizer

# from torchvision.transforms import *


def deep_reid(query_np, gal_np, model):
    # print ("reid")
    # model params  
    use_gpus = False
    # root = REID_PATH + "/data"
    # name = "single_test"
    # split_id = 0
    # height = 256
    # width = 128
    # train_batch_size = 32
    # test_batch_size = 1
    # workers = 4
    # train_sampler = ''
    # num_instances = 4
    # cuhk03_labeled = False
    # cuhk03_classic_split = False

    # torch.manual_seed(1)
    path_to_weights =  REID_PATH + "/weights/resnet50_market_xent/resnet50_market_xent.pth.tar"
    save_log_dir =  REID_PATH + "/log/eval-resnet50"

    # log_name = 'log_test.txt'
    # sys.stdout = Logger(osp.join(save_log_dir, log_name))

    # Initialize Data
    # dm = ImageDataManager(use_gpus, [name], [name], root, split_id, height, width, train_batch_size, test_batch_size, workers, train_sampler, num_instances, cuhk03_labeled, cuhk03_classic_split)
    # testloader_dict = dm.return_dataloaders()

    # Initialize Model
    # st = time.time()
    # model = models.init_model("resnet50", num_classes=1, loss={'xent'}, use_gpu=use_gpus)
    # et = time.time()
    # print("reid time", et-st)
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
    # queryloader = testloader_dict[name]['query']
    # galleryloader = testloader_dict[name]['gallery']
    # distmat = test(model, query_np, gal_np, use_gpus) 
    #visualize_ranked_results(distmat, dm.return_testdataset_by_name(name), save_log_dir + "/ranked_results/yeet", 1)
    #print ("DIST:", distmat)
    #gallery = dm.return_testdataset_by_name(name)[1]
    return test(model, query_np, gal_np, use_gpus) 

def t1(img): # opencv
    st = time.time()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize = torch_trans.Normalize(mean=imagenet_mean, std=imagenet_std)
    transforms = []
    transforms += [torch_trans.Resize((256, 128))]
    transforms += [torch_trans.ToTensor()]
    transforms += [normalize]
    transforms = torch_trans.Compose(transforms)
    # img = cv2.resize(img, (128, 256), interpolation=cv2.INTER_NEAREST)
    img = Image.fromarray(img)
    img = transforms(img)
    et = time.time()
    # print("cv time",et-st)
    return img

def t2(img_path=None): # PIL
    st = time.time()
    if img_path is not None:
        img = Image.open(img_path)
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        normalize = Normalize(mean=imagenet_mean, std=imagenet_std)
        
        transforms = []
        transforms += [Resize((256, 128))]
        transforms += [ToTensor()]
        transforms += [normalize]

        transforms = Compose(transforms)

        img = transforms(img)
        et = time.time()
        print("PIL time",et-st)
        return img

# def main_test():
#     img = cv2.imread("naturo-monkey-selfie.jpg", cv2.IMREAD_COLOR)
#     a = t1(img)
#     b = t2("naturo-monkey-selfie.jpg")
#     print ("opencv",a)
#     print ("PIL", b)
#     return torch.eq(a, b).all()

def test(model, query_np, gal_np, use_gpu, ranks=[1, 5, 15, 20], return_distmat=True): # !!!!!!
    # batch_time = AverageMeter()
    
    model.eval()
    # imgs -> queryloader / galleryloader -> testloader_dict[q/g] -> dm.return_dataloaders() -> imagedatamanager (line 57) -> transform function -> img dataset thingy -> dataset loaders ->  somehow transfroms img from img path -> transformed image. 
    # we want to stop image writing from (numpy array -> )
    with torch.no_grad():
        qf = []
        query_np = t1(query_np)
        query_np = query_np.unsqueeze(0)
        # print ("T1",query_np.size())
        # query_np = t2("hi.jpg")
        # print ("T2", query_np)
        features = model(query_np) # !!!
        features = features.data.cpu()
        qf.append(features)
        # q_pids.extend(pids)
        # q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        # q_pids = np.asarray(q_pids)
        # q_camids = snp.asarray(q_camids)

        # #print ("\n")
        # #print ("pid", q_pids)
        # #print ("camid", q_camids)
        # #print ("\n")
        gf = []
        # #print ("\n")
        # #print ("g_pid", pids)
        # #print ("g_camid", camids) # 
        gal_np = t1(gal_np)
        gal_np = gal_np.unsqueeze(0)
        features = model(gal_np)
        # #print ("FEaT1", features)
        # #print ("FL1", len(features[0]))

        features = features.data.cpu()
        # #print ("FEaT", features)
        # #print ("FL", len(features[0]))
        gf.append(features)
        # g_pids.extend(pids)
        # g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        # g_pids = np.asarray(g_pids)
        # g_camids = np.asarray(g_camids)

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
# img = cv2.imread("hi.jpg")
# deep_reid(img, img)