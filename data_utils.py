from os import listdir, path
from os.path import join

import cv2
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

import utility


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


def load_hr_image_auto_decompose(img, scale):
    hr = cv2.imread(img)
    region_size = utility.get_max_region_size(hr.shape[0] / scale, hr.shape[1] / scale)
    regions = utility.image_decomposition(hr, region_size)
    transform = Compose([
        ToPILImage(),
        Resize(region_size / scale, interpolation=Image.BICUBIC),
        ToTensor()
    ])
    return transform, regions


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.images = []
        self.hr_transform = Compose([
            ToPILImage(),
            ToTensor()
        ])
        for x in listdir(dataset_dir):
            x = path.join(dataset_dir, x)  # Fix python broken listdir
            lr_transform, regions = load_hr_image_auto_decompose(x, upscale_factor)
            for i in range(0, len(regions)):
                self.images.append([regions[i], lr_transform])

    def __getitem__(self, index):
        pythonisapeaceofshit = self.images[index]
        hr = pythonisapeaceofshit[0]
        transform = pythonisapeaceofshit[1]
        lr = transform(hr)
        return lr, self.hr_transform(hr)

    def __len__(self):
        return len(self.images)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.images = []
        self.hr_transform = Compose([
            ToPILImage(),
            ToTensor()
        ])
        for x in listdir(dataset_dir):
            x = path.join(dataset_dir, x)  # Fix python broken listdir
            lr_transform, regions = load_hr_image_auto_decompose(x, upscale_factor)
            for i in range(0, len(regions)):
                self.images.append([regions[i], lr_transform])

    def __getitem__(self, index):
        pythonisapeaceofshit = self.images[index]
        hr = pythonisapeaceofshit[0]
        transform = pythonisapeaceofshit[1]
        lr = transform(hr)
        return lr, self.hr_transform(hr)

    def __len__(self):
        return len(self.images)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
