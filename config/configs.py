import os

IMG_HEIGHT = 256
IMG_WIDTH = 256


class Config:
    def __init__(self):
        self.scribbler = False
        self.folder = 'C:\\Users\\Public\\Desktop\\Tesis\\Models\\Pix2Pix\\'
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
        self.checkpoint_dir = self.folder + 'training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
