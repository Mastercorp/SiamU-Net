import glob
from torch.utils import data
from PIL import Image
import torchvision
import random
from module import dataaug


class ISBIDataset(data.Dataset):

    def __init__(self, gloob_dir_train, gloob_dir_label, eval, augment):
        self.gloob_dir_train = gloob_dir_train
        self.gloob_dir_label = gloob_dir_label
        self.crop_nopad = torchvision.transforms.CenterCrop(324)
        self.eval = eval
        self.augment = augment
        self.changetotensor = torchvision.transforms.ToTensor()

        self.trainfiles = sorted(glob.glob(self.gloob_dir_train),
                                 key=lambda name: int(name[self.gloob_dir_train.rfind('*'):
                                                           -(len(self.gloob_dir_train) - self.gloob_dir_train.rfind(
                                                               '.'))]))

        self.labelfiles = sorted(glob.glob(self.gloob_dir_label),
                                 key=lambda name: int(name[self.gloob_dir_label.rfind('*'):
                                                           -(len(self.gloob_dir_label) - self.gloob_dir_label.rfind(
                                                               '.'))]))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.trainfiles)

    def __getitem__(self, index):
        'Generates one sample of data'
        # files are sorted depending the last number in their filename
        # for example : "./ISBI 2012/Train-Volume/train-volume-*.tif"

        img_path = self.trainfiles[index]
        label_path = self.labelfiles[index]
        img = Image.open(img_path).convert("L")
        label = Image.open(label_path).convert("L")
        img_name = img_path.rsplit("/", 1)[1].split(".")[0]

        if not self.eval:

            if random.random() < self.augment["angle"]:
                angle = random.randint(0, self.augment["angleparts"] - 1) * (360.0 / self.augment["angleparts"])
                img = dataaug.rotate_img(img, angle)
                label = dataaug.rotate_img(label, angle)
                label = dataaug.maplabel(label, 127, 0, 255)

            if random.random() < self.augment["flipping"]:
                label = dataaug.flip_lr(label)
                img = dataaug.flip_lr(img)

            if random.random() < 0:
                label = dataaug.flip_tb(label)
                img = dataaug.flip_tb(img)

            if random.random() < self.augment["dataaug"]:

                sigma = random.randint(self.augment["sigmamin"], self.augment["sigmamax"])
                # to shift in gradient direction -, else it shifts in the other direction
                scalex = random.randint(self.augment["minscale"], self.augment["maxscale"])
                scaley = random.randint(self.augment["minscale"], self.augment["maxscale"])
                if random.random() < 0.5:
                    scalex = scalex * -1
                if random.random() < 0.5:
                    scaley = scaley * -1
                img, indices = dataaug.gradtrans(img, scalex, scaley, sigma)

                # map label to the same coordinates as img
                label = dataaug.indextrans(label, indices)
                # always use maplabel to convert labels to 0 and 255!!
                label = dataaug.maplabel(label, 127, 0, 255)

        label = self.changetotensor(label)
        img = self.changetotensor(img)

        return img, label, img_name

