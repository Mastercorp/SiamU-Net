import glob
from torch.utils import data
from PIL import Image
import torchvision
import random
from module import dataaug
import os


class ISBIDataset(data.Dataset):

    def __init__(self, gloob_dir_train, gloob_dir_label, eval, augment):
        self.gloob_dir_train = gloob_dir_train
        self.gloob_dir_label = gloob_dir_label
        self.crop_nopad = torchvision.transforms.CenterCrop(324)
        self.eval = eval
        self.augment = augment
        self.changetotensor = torchvision.transforms.ToTensor()

        self.labelfiles = sorted(glob.glob(self.gloob_dir_label),
                                 key=lambda name: int(name[self.gloob_dir_label.rfind('*'):
                                                           -(len(self.gloob_dir_label) - self.gloob_dir_label.rfind(
                                                               '.'))]))

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.labelfiles)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # files are sorted depending the last number in their filename
        # for example : "./ISBI 2012/Train-Volume/train-volume-*.tif"

        label_path1 = self.labelfiles[index]
        label1 = Image.open(label_path1).convert("L")
        label_name1 = label_path1.rsplit("/", 1)[1].split(".")[0]

        # number present in imagename: train-volume-21.tif => 21
        imgnr1 = int(label_name1.rsplit("-", 1)[1])

        img_path1 = self.gloob_dir_train.format(imgnr1)
        img_dir1 = img_path1.rsplit("/", 1)[0]

        img1 = Image.open(img_path1).convert("L")

        img_name1 = img_path1.rsplit("/", 1)[1].split(".")[0]
        img_format1 = img_path1.rsplit("/", 1)[1].split(".")[1]

        #search backward first, if not found, search forward
        #else take the same image
        img_name2 = img_name1.rsplit("-", 1)[0] + "-" + str(imgnr1 - 1)
        img_path2 = img_dir1 + "/" + img_name2 + "." + img_format1
        exists = os.path.isfile(img_path2)

        # if a following image exists, use it
        # else, check if a previous image exists
        # if not, take the same image
        if not exists:
            img_name2 = img_name1.rsplit("-", 1)[0] + "-" + str(imgnr1 + 1)
            img_path2 = img_dir1 + "/" + img_name2 + "." + img_format1
            exists = os.path.isfile(img_path2)

        if exists:
            img2 = Image.open(img_path2).convert("L")
        else:
            img2 = img1

        if not self.eval:
            if random.random() < self.augment["angle"]:
                angle = random.randint(0, self.augment["angleparts"] - 1) * (360.0 / self.augment["angleparts"])
                img1 = dataaug.rotate_img(img1, angle)
                img2 = dataaug.rotate_img(img2, angle)

                label1 = dataaug.rotate_img(label1, angle)
                label1 = dataaug.maplabel(label1, 127, 0, 255)

            if random.random() < self.augment["flipping"]:
                label1 = dataaug.flip_lr(label1)
                img1 = dataaug.flip_lr(img1)
                img2 = dataaug.flip_lr(img2)

            if random.random() < self.augment["dataaug"]:

                sigma = random.randint(self.augment["sigmamin"], self.augment["sigmamax"])
                # to shift in gradient direction -, else it shifts in the other direction
                scalex = random.randint(self.augment["minscale"], self.augment["maxscale"])
                scaley = random.randint(self.augment["minscale"], self.augment["maxscale"])
                if random.random() < 0.5:
                    scalex = scalex * -1
                if random.random() < 0.5:
                    scaley = scaley * -1

                img1, indices = dataaug.gradtrans(img1, scalex, scaley, sigma)
                # map trainimg2 to the same coordinates as trainimg1
                img2 = dataaug.indextrans(img2, indices)

                label1 = dataaug.indextrans(label1, indices)
                label1 = dataaug.maplabel(label1, 127, 0, 255)

        img = dataaug.combine_img(img1, img2)
        trainlabel = self.changetotensor(label1)
        trainimg = self.changetotensor(img)

        return trainimg, trainlabel, img_name1
