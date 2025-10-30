import os 
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset 

DATA_DIR = r"/mnt/c/Users/jwiso/OneDrive/Desktop/COMP3710/keras_png_slices_data/keras_png_slices_data"

TEST_DIR_SEG     = os.path.join(DATA_DIR, "keras_png_slices_seg_test")
TRAIN_DIR_SEG    = os.path.join(DATA_DIR, "keras_png_slices_seg_train")
VALIDATE_DIR_SEG = os.path.join(DATA_DIR, "keras_png_slices_seg_validate")

TEST_DIR     = os.path.join(DATA_DIR, "keras_png_slices_test")
TRAIN_DIR    = os.path.join(DATA_DIR, "keras_png_slices_train")
VALIDATE_DIR = os.path.join(DATA_DIR, "keras_png_slices_validate")  # <-- was wrong before

print("Loading data...")

def _list_pngs(p):
    return sorted([f for f in os.listdir(p) if f.lower().endswith(".png")])

class OASISProjectDataset(Dataset):
    def __init__(self, img_directory: str, mask_directory: str):
        self.img_directory, self.mask_directory = img_directory, mask_directory
        self.image_names = _list_pngs(img_directory)
        self.mask_names  = _list_pngs(mask_directory)
        self.len = len(self.image_names)
        if len(self.mask_names) != self.len:
            raise ValueError(f"Image/mask count mismatch: {self.len} vs {len(self.mask_names)}")
        # Optional: show a couple so you can sanity-check pairing:
        print(f"{os.path.basename(img_directory)}: {self.len} imgs | "
              f"{os.path.basename(mask_directory)}: {len(self.mask_names)} masks")
        if self.len:
            print(" e.g.", self.image_names[0], "<->", self.mask_names[0])

    def __len__(self):
        return self.len

    def __getitem__(self, index: int):
        img_location  = os.path.join(self.img_directory,  self.image_names[index])
        mask_location = os.path.join(self.mask_directory, self.mask_names[index])
        # T.ToTensor and PILToTensor are *callables* — call them, and force grayscale.
        img  = T.ToTensor()(    Image.open(img_location).convert("L"))                # [1,H,W], float in [0,1]
        mask = T.PILToTensor()( Image.open(mask_location).convert("L")).float()      # [1,H,W], 0..255
        mask = (mask > 0).float()                                                    # binarise to {0,1}
        return img, mask

def get_datasets_and_data_loaders(batch_size: int):
    training_data_set   = OASISProjectDataset(TRAIN_DIR,    TRAIN_DIR_SEG)
    training_data_loader= DataLoader(training_data_set, batch_size=batch_size, shuffle=True)

    test_data_set       = OASISProjectDataset(TEST_DIR,     TEST_DIR_SEG)
    test_data_loader    = DataLoader(test_data_set, batch_size=batch_size)

    validation_data_set = OASISProjectDataset(VALIDATE_DIR, VALIDATE_DIR_SEG)
    validation_data_loader = DataLoader(validation_data_set, batch_size=batch_size)

    return training_data_set, training_data_loader, test_data_set, test_data_loader, validation_data_set, validation_data_loader
