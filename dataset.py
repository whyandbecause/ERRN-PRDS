import os

from torch.utils.data import Dataset
from PIL import Image


class RainBackImage(Dataset):
    def __init__(self, path_rain, path_gt, transform=None):

        self.transform = transform
        self.rain_images = []
        self.ground_images = []
        for index, filename in enumerate(os.listdir(path_rain)):

            # if dir == 'rainy_image':
            #     list.sort(key=lambda x: (int(x.split('.')[0].split('_')[0]), int(x.split('.')[0].split('_')[1])))
            # else:
            #     list.sort(key=lambda x: int(x.split('.')[0]))
            # for filename in list:
            #     if index == 0:
            self.rain_images.append(os.path.join(path_rain, filename))
        for index, filename in enumerate(os.listdir(path_gt)):
            self.ground_images.append( os.path.join(path_gt, filename))

    def __len__(self):
        return len(self.rain_images)

    def __getitem__(self, index):
        rain_path= self.rain_images[index]
        ground_path= self.ground_images[index]
        rain_image = Image.open(rain_path)
        ground_image = Image.open(ground_path)
        rain_image = rain_image.convert('RGB')
        ground_image = ground_image.convert('RGB')
        if self.transform:
            rain_image = self.transform(rain_image)
            ground_image = self.transform(ground_image)
        return rain_image, ground_image, rain_path.split('\\')[-1]

# class TestDatasets(Dataset):
#     def __init__(self, path, transform=None):
#
#         self.root = path
#         self.transform = transform
#         self.rain_images = []
#         self.ground_images = []
#         for index, dir in enumerate(os.listdir(self.root)):
#             list = os.listdir(os.path.join(self.root, dir))
#             if dir == 'rainy_image':
#                 list.sort(key=lambda x: (int(x.split('.')[0].split('_')[0]), int(x.split('.')[0].split('_')[1])))
#             else:
#                 list.sort(key=lambda x: int(x.split('.')[0]))
#             for filename in list:
#                 if index == 0:
#                     self.ground_images.append(os.path.join(self.root, dir, filename))
#                 else:
#                     self.rain_images.append(os.path.join(self.root, dir, filename))
#
#     def __len__(self):
#         return len(self.rain_images)
#
#     def __getitem__(self, index):
#         rain_path= self.rain_images[index]
#         ground_path= self.ground_images[index]
#         rain_image = Image.open(rain_path)
#         ground_image = Image.open(ground_path)
#         rain_image = rain_image.convert('RGB')
#         ground_image = ground_image.convert('RGB')
#         if self.transform:
#             rain_image = self.transform(rain_image)
#             ground_image = self.transform(ground_image)
#         return rain_image, ground_image, rain_path.split('/')[-1]


