import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tool import custom_transforms as tr

class Stage1_InferDataset(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        self.object = self.path_label()

    def __getitem__(self, index):
        fn = self.object[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return fn.split('/')[-1][:-4], img
        
    def __len__(self):
        return len(self.object)
        
    def path_label(self):
        path_list = []
        for root, dirname, filename in os.walk(self.data_path):
            for f in filename:
                image_path = os.path.join(root, f)
                path_list.append(image_path)
        return path_list

class Stage1_TrainDataset(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None, dataset=None):
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset
        self.object = self.path_label()
        

    def __getitem__(self, index):
        fn, label = self.object[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return fn.split('/')[-1][:-4], img, label
        
    def __len__(self):
        return len(self.object)
        
    def path_label(self):
        path_label = []
        for root, dirname, filename in os.walk(self.data_path):
            for f in filename:
                image_path = os.path.join(root, f)
                fname = f[:-4]
                label_str = fname.split(']')[0].split('[')[-1]
                if self.dataset == 'digestpath':
                    image_label = torch.Tensor([int(label_str[0]), int(label_str[2])])
                elif self.dataset == 'wien':
                    image_label = torch.Tensor([int(label_str[0]), int(label_str[2])])
                elif self.dataset == 'hph':
                    image_label = torch.Tensor([int(label_str[0]), int(label_str[2])])
                elif self.dataset == 'glas':
                    image_label = torch.Tensor([int(label_str[0]), int(label_str[3])])
                path_label.append((image_path, image_label))
        return path_label


class Stage2_Dataset(Dataset):
    def __init__(self, args, base_dir, split):

        super().__init__()
        self._base_dir = base_dir
        self.split = split

        if self.split   == "train":
            self._image_dir     = os.path.join(self._base_dir, 'train/img/')
            self._cat_dir       = os.path.join(self._base_dir, 'train_PM_IBDR/')
        elif self.split == 'val':
            self._image_dir = os.path.join(self._base_dir, 'val/img/')
            self._cat_dir   = os.path.join(self._base_dir, 'val/mask/')
        elif self.split == 'test':
            self._image_dir = os.path.join(self._base_dir, 'test/img/')
            self._cat_dir   = os.path.join(self._base_dir, 'test/mask/')
        elif self.split == 'pred':
            self._image_dir = os.path.join(self._base_dir,'test/img/')
            self._cat_dir   = os.path.join(self._base_dir, 'test/mask/')
        self.args = args
        self.filenames = [os.path.splitext(file)[0] for file in os.listdir(self._image_dir) if not file.startswith('.')]
        self.images = [os.path.join(self._image_dir, fn + '.png') for fn in self.filenames]
        self.categories = [os.path.join(self._cat_dir, fn + '.png') for fn in self.filenames]

        assert (len(self.images) == len(self.categories))
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.split == "train":
            _img, _target = self._make_img_gt_point_pair(index)
            sample = {'image': _img, 'label': _target}
        elif (self.split == 'val') or (self.split == 'test'):
            _img, _target, = self._make_img_gt_point_pair(index)
            sample = {'image': _img, 'label': _target}
            image_dir = self.images[index]
        elif (self.split == 'pred'):
            _img, _target, = self._make_img_gt_point_pair(index)
            sample = {'image': _img, 'label': _target}
            image_dir = self.images[index]
        if self.split == "train":
            return self.transform_tr_ab(sample)
        elif (self.split == 'val') or (self.split == 'test'):
            return self.transform_val(sample), image_dir
        elif self.split == 'pred':
            return self.transform_val(sample),image_dir
    def _make_img_gt_point_pair(self, index):
        if self.split == "train": 
            _img = Image.open(self.images[index]).convert('RGB')
            _target = Image.open(self.categories[index])
            return _img, _target
        elif (self.split == 'val') or (self.split == 'test'):
            _img = Image.open(self.images[index]).convert('RGB')
            _target = Image.open(self.categories[index])
            return _img,_target
        elif self.split == 'pred':
            _img = Image.open(self.images[index]).convert('RGB')
            _target = Image.open(self.categories[index])
            return _img,_target

    def transform_tr_ab(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip_ab(),
            tr.RandomVerticalFlip_ab(),
            tr.Normalize_ab(),
            tr.ToTensor_ab()   
        ])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.Normalize(),
            tr.ToTensor()])
        return composed_transforms(sample)

    def __str__(self):
        return None

def make_data_loader(args, **kwargs):

    train_set   = Stage2_Dataset(args, base_dir=args.dataroot, split='train')
    val_set     = Stage2_Dataset(args, base_dir=args.dataroot, split='val')
    test_set    = Stage2_Dataset(args, base_dir=args.dataroot, split='test')


    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader
