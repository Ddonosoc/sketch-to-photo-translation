import os

IMG_HEIGHT = 256
IMG_WIDTH = 256


class Config:
    def __init__(self, checkpoint_name='training_checkpoints'):
        self.scribbler = False
        self.folder = 'C:\\Users\\Public\\Desktop\\Tesis\\Models\\Quimera\\'
        self.dataset_name = "shoes"
        self.folder_dataset_train = 'C:/Users/Public/Desktop/Tesis/Databases/datasets/pix2pixDafitiQuickdraw/train/'
        self.folder_dataset_test = 'C:/Users/Public/Desktop/Tesis/Models/Evaluation/ResNet50/experiments_results/testBC/'
        self.dataset_foldername = "C:\\Users\\Public\\Desktop\\Tesis\\Models\\Evaluation\\ResNet50\\experiments_results\\testC\\testC\\"
        self.symbol_replacement = "\\"
        self.folder_dest = self.folder + "results" + self.symbol_replacement
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.OUTPUT_CHANNELS = 3
        self.LAMBDA = 100
        self.TV_WEIGHT = 10
        self.F_WEIGHT = 100
        self.BUFFER_SIZE = 400
        self.BATCH_SIZE = 1
        self.checkpoint_dir = self.folder + checkpoint_name
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.pix2pix = None
        # self.scribbler = False
        # self.folder = '/users/ddonoso/Models/Quimera/'
        # self.dataset_name = "shoes"
        # self.folder_dataset_train = '/users/ddonoso/Models/pix2pixDafitiQuickdraw/train/'
        # self.folder_dataset_test = '/users/ddonoso/Models/pix2pixDafitiQuickdraw/test/'
        # self.dataset_foldername = "/users/ddonoso/Models/testC/testC/"
        # self.symbol_replacement = "/"
        # self.folder_dest = self.folder + "results" + self.symbol_replacement
        # self.IMG_HEIGHT = IMG_HEIGHT
        # self.IMG_WIDTH = IMG_WIDTH
        # self.OUTPUT_CHANNELS = 3
        # self.LAMBDA = 100
        # self.TV_WEIGHT = 0.0005
        # self.F_WEIGHT = 0.05
        # self.BUFFER_SIZE = 400
        # self.BATCH_SIZE = 1
        # self.checkpoint_dir = self.folder + checkpoint_name
        # self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.epochs = 200
        self.epoch_decay = 100
        self.lr = 0.0002
        self.beta_1 = 0.5
        self.len_dataset = 7000
        self.loss_mode = 'lsgan'
        self.cycle_loss_weight = 10.0
        self.identity_loss_weight = 0.5
        self.gradient_penalty_mode = 'none'
        self.gradient_penalty_weight = 10.0
