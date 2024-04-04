# 
#  Copyright (C) 2023 Istituto Italiano di Tecnologia (IIT)
#  
#  This software may be modified and distributed under the terms of the
#  GPL-2+ license. See the accompanying LICENSE file for details.
#  
from digit_interface.digit import Digit
import configparser
import numpy as np
import os
from PIL import Image
file_path = os.path.abspath(__file__)
from surfaceclassifier.src_compare_dataset.composed_model import ComposedModel
from surfaceclassifier.src_compare_dataset.image_dataset import  get_test_transform_class
from accelerate import Accelerator
import yaml
from fire import Fire
import torch
import cv2
from PIL import Image


from pathlib import Path
def load_hparms(path: str) -> dict:
    """ Loads the hparms from a yaml file and returns a dict. """

    with open(path, "r") as f: 
        hparms = yaml.full_load(f)

    return hparms

@torch.inference_mode()
class PrunePointCloud():
    """
    A class to prune a point cloud depending on an input image.
    
    '''

    Attributes
    ----------
    device : torch.device
        the device where the model is stored
    poses_array : np.array
        numpy array containing the poses of the points
    threshold : float
        float that sets the threshold for the pruning
    trasnforms : torchvision.transforms.Compose
        set of transformations for the images
    Methods
    -------
    get_point_cloud(digit_image):
        Prune the point cloud based on the input image.
    
    """

    def __init__(self, 
                conf_file,
                ckpt,
                config_file_path=os.path.join(file_path.rsplit("/", 1)[0], 'config', 'config.ini')):
        """
        Constructor.

        Parameters
        ----------
        config_file_path : str, optional
            path of the config file to be parsed
        
        """

        # Load the config.ini file
        print(config_file_path)
        config = configparser.ConfigParser()
        config.read(config_file_path)

        # Parse the file
        paths = config['Paths']
        # # Parse the paths

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Get the latent vector of the background
        self.background = Image.open(paths['background'])

        accelerator = Accelerator(split_batches=True)
        # Load config
        accelerator.print("Loading hparams...")

        hparams = load_hparms(conf_file)
        hparams_model = hparams["model"]
        hparams_data  = hparams["data"]

        # 2) model
        accelerator.print("Loading model...")

        self.model = ComposedModel(**hparams_model)

        ckpt_state_dict = torch.load(ckpt, map_location="cpu")["model"]
        self.model.load_state_dict(ckpt_state_dict, strict=True)

        self.model = accelerator.prepare(self.model)
        self.transform = get_test_transform_class(**hparams_data)
        self.model.cuda()
        self.model.eval()
        self.add = 0

        bck = np.array(self.background)-np.array(self.background)
        bck += 127
        bck = self.transform(Image.fromarray(bck))
        bck = bck.cuda()
        with torch.no_grad():
            logits = self.model.forward(bck.unsqueeze(0))
        self.background_features = logits['backbone_features']


    def get_point_cloud(self, digit_image):

        digit_image -= self.background
        digit_image += 127

        digit_image = self.transform(Image.fromarray(digit_image))
        digit_image = digit_image.cuda()
        with torch.no_grad():

            logits = self.model.forward(digit_image.unsqueeze(0))

        print(torch.argmax(logits['logits']).item())
        class_predicted = torch.argmax(logits['logits']).item()

        if torch.nn.CosineSimilarity()(logits['backbone_features'], self.background_features) > 0.415:
            return np.empty((0,3))

        return self.poses_array[np.where(self.labels==class_predicted)]

    def get_class(self, digit_image):

        digit_image -= self.background
        digit_image += 127

        digit_image = self.transform(Image.fromarray(digit_image))
        digit_image = digit_image.cuda()
        with torch.no_grad():
            logits = self.model.forward(digit_image.unsqueeze(0))

        class_predicted = torch.argmax(logits['logits']).item()

        return int(class_predicted)
    

    def get_class_live(self, digit):


        while True:
            digit_image = digit.get_frame()
            digit_image = cv2.cvtColor(digit_image, cv2.COLOR_BGR2RGB)
            digit_image -= self.background
            digit_image += 127
            # cv2.imshow('ciao', digit_image)
            # cv2.waitKey(1)
            # point_clouds_array = np.empty((0,3))

            digit_image = self.transform(Image.fromarray(digit_image))
            digit_image = digit_image.cuda()
            with torch.no_grad():
                # print(digit_image.unsqueeze(0).shape)
                logits = self.model.forward(digit_image.unsqueeze(0))

            # class = logits.
            print(torch.argmax(logits['logits']).item())
    
def main(config: str, ckpt: str, data_path:str):
    prune = PrunePointCloud(config, ckpt)

if __name__ == "__main__":
    Fire(main)
