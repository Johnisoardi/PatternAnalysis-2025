import os 
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset 


DATA_DIR = r"/mnt/c/Users/jwiso/OneDrive/Desktop/COMP3710/keras_png_slices_data/keras_png_slices_data"
TEST_DIR_SEG = DATA_DIR + "keras_png_slices_seg_test/"
TRAIN_DIR_SEG = DATA_DIR + "keras_png_slices_seg_train/"
VALIDATE_DIR_SEG = DATA_DIR + "keras_png_slices_seg_validate/"
TEST_DIR = DATA_DIR + "keras_png_slices_test/"
TRAIN_DIR = DATA_DIR + "keras_png_slices_train/"
VALIDATE_DIR = DATA_DIR + "keras_png_slices_seg_validate/"
VALIDATE_DIR = DATA_DIR + "keras_png_slices_seg_validate/"

print("Loading data...")

class OASISProjectDataset(Dataset):
    """
    Dataset class specific to the Oasis Dataset.
    """
    def __init__(self, img_directory: str, mask_directory: str):
        """
        Initialises an OasisProjectDataset Object. 
        precondition: Data has names as downloaded from rangpur. 
        """
        # Since original image and mask folders are the same size and have same naming convention, we can 
        # just match images to masks via sorting both lists!
        self.img_directory, self.mask_directory = img_directory, mask_directory
        self.image_names = sorted(os.listdir(img_directory))
        self.mask_names = sorted(os.listdir(mask_directory)) 
        self.len = len(self.image_names)

        if len(self.mask_names) != self.len:
            raise ValueError()

    def __len__(self):
        return self.len

    def __getitem__(self, index: int):
        img_location = os.path.join(self.img_directory, self.image_names[index])
        mask_location = os.path.join(self.mask_directory, self.mask_names[index]) 
        # Need channel dimension
        img, mask = T.ToTensor(Image.open(img_location)), (T.ToTensor(Image.open(mask_location))> 0.5).float() # convert masks to boolean data
        return img, mask

def get_datasets_and_data_loaders(batch_size: int):
    """
    batch_size: the batch size for the data loaders

    """
    training_data_set = OASISProjectDataset(TRAIN_DIR, TRAIN_DIR_SEG)
    training_data_loader = DataLoader(training_data_set, batch_size = batch_size, shuffle = True) # best to shuffle training data
    test_data_set = OASISProjectDataset(TEST_DIR, TEST_DIR_SEG)
    test_data_loader = DataLoader(test_data_set, batch_size = batch_size)
    validation_data_set = OASISProjectDataset(VALIDATE_DIR, VALIDATE_DIR_SEG)
    validation_data_loader = DataLoader(validation_data_set, batch_size = batch_size)
    return training_data_set, training_data_loader, test_data_set, test_data_loader, validation_data_set,validation_data_loader