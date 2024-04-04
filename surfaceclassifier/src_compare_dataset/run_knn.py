from image_dataset import ImageFolderIdx, get_test_transform, get_test_transform_bw
from feature_extractor import FeatureExtractor

from os.path import join
from fire import Fire

from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA



import torch.nn.functional as F

from pathlib import Path


@torch.inference_mode()
def extract_features(dataset, model, batch_size=64):

    model.cuda()
    model.eval()

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    features_bank = None
    labels_bank = None
    flatten = nn.Flatten()

    for i, batch in enumerate(loader):

        print(f"Extracting features: {i+1}/{len(loader)}")

        images, idxs, labels = batch["images"], batch["idxs"], batch["labels"] 

        features = model(images.cuda())
        features = flatten(features)

        if features_bank is None:
            features_bank = torch.zeros([len(dataset), features.shape[1]]).cuda()
            labels_bank = -torch.ones([len(dataset)], dtype=torch.long).cuda()

        features_bank[idxs.cuda()] = features
        labels_bank[idxs.cuda()]= labels.cuda()

    return features_bank, labels_bank


#  functions to print tensors into csv
def _row_to_csv(row):
    if isinstance(row, torch.Tensor): row = row.cpu()
    cvt_fnc = lambda x: str(x.item())
    return ",".join(list(map(cvt_fnc, row)))

def _tensor_to_csv(tensor):
    tensor = tensor.cpu()
    final_str = ""
    for row in tensor:
        final_str += _row_to_csv(row)
        final_str += "\n"

    return final_str


def KNN(X, Y, Xt, Yt, 
        out_dir: str,
        ths: list = [0.1, 0.3, 0.5, 0.8, 0.9], 
        pca_components: int = 10,
        ks: list = [1, 3, 5, 7, 9, 11]):

    Path(out_dir).mkdir(exist_ok=True, parents=True)

    # apply PCA if needed
    if pca_components > 0:
        pca = PCA(n_components=pca_components, whiten=False)
        X  = pca.fit_transform(X.cpu())
        Xt = pca.transform(Xt.cpu())
        X = torch.tensor(X).cuda()
        Xt = torch.tensor(Xt).cuda()

    # normalize to compute cosine similarity
    X  = F.normalize(X, dim=1)
    Xt = F.normalize(Xt, dim=1)

    # compute cosine similarity
    similarities  = 1 - 0.5 * torch.cdist(Xt, X)

    # save similarities
    with open(join(out_dir, "similarities.csv"), "w+") as f:
        print(_tensor_to_csv(similarities), file=f)


    #  save results for any threshold 
    for th in ths:

        output_file = join(out_dir, f"th_{int(th*100)}.csv")

        # get similarities that are more than current threshold
        output = []
        for row in similarities:
            output.append(list(torch.where(row>th)[0].cpu()))

        with open(output_file , "w+") as f:
            for row in output:
                print(_row_to_csv(row), file=f)

    # save accuracies for any k
    for k in ks:
        _, indices = torch.topk(similarities, k=k, largest=True)
        Ypred, _ = torch.mode(Y[indices])
        accuracy = torch.sum(Ypred == Yt)/len(Yt)

        with open(join(out_dir, "accuracies.txt"), "a+") as f:
            print(f"k={k}, accuracy={accuracy:4f}", file=f)


def load_labels_from_csv(csv_path):

    with open(csv_path, "r") as f:
        lines = [l.split(",") for l in f.readlines()]
        
    return [int(class_n.strip()) for _, class_n in lines]
        


def main(model_name: str,
         out_dir: str,
         dataset_database: str = None,
         features_database: str = None,
         dataset_test: str = None, 
         resize_size: Tuple = (320, 240), 
         crop_size: Tuple = (320, 240),
         bw_transform: bool = False,
         custom_ckpt: Optional[str] = None, 
         batch_size: int = 32,
         ths: list = [0.1, 0.3, 0.5, 0.8, 0.9],
         ks: list = [1,3,5,7,9,11,13], 
         pca_components: int = 10):

    # get model
    feature_extractor = FeatureExtractor(name=model_name, custom_ckpt=custom_ckpt).cuda()

    # get transform
    if bw_transform:
        print("Using BW transform")
        transform = get_test_transform_bw(resize_size=resize_size, crop_size=crop_size)
    else:
        print("Using Color transform")
        transform = get_test_transform(resize_size=resize_size, crop_size=crop_size)
    

    # extract or load features/labels of the database
    if features_database:
        print(f"LOADING database features from folder: {features_database}")
        X_db = torch.load(join(features_database, "features.tensor"))
        Y_db = load_labels_from_csv(join(features_database, "labels.csv"))
    else:
        print(f"EXTRACTING database features from folder: {dataset_database}")
        dataset_database = ImageFolderIdx(root=dataset_database, 
                                        transform=transform, 
                                        csv_labels=join(dataset_database, "labels.csv"))
        X_db, Y_db = extract_features(dataset_database, feature_extractor, batch_size)
    

    # extract features/labels of the test set
    print(f"EXTRACTING test features from folder: {dataset_test}")
    dataset_test = ImageFolderIdx(root=dataset_test, 
                                  transform=transform, 
                                  csv_labels=join(dataset_test, "labels.csv"))
    

    Xt, Yt = extract_features(dataset_test, feature_extractor, batch_size)

    KNN(X_db, Y_db, Xt, Yt, 
        out_dir=out_dir, 
        ths=ths, 
        pca_components=pca_components,
        ks=ks)
    
if __name__ == "__main__":
    Fire(main)

