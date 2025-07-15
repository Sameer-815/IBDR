import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from tool.GenDataset import make_data_loader
from tool.lr_scheduler import LR_Scheduler
from tool.metrics import Evaluator
from tool.IICB import IICBLoss
from network.DPR_Net import Net

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.writer = SummaryWriter(log_dir='logs/')
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader  = make_data_loader(args, **kwargs)
        self.nclass = args.n_class
        model = Net(encoder_name='timm-resnest200e',encoder_weights='imagenet',in_channels=3,classes=self.nclass)
        lr = self.args.lr
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
        self.criterion = IICBLoss(alpha=1, beta=0.2, gamma=0.1, delta=0.1, eta=0.1, tau=0.4)
        self.loss_weight_scheduler = AdaptiveWeightScheduler(total_epochs=args.epochs)
        self.model, self.optimizer = model, optimizer
        self.evaluator = Evaluator(self.nclass)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))
        self.best_pred = 0.0

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        self.model = self.model.cuda()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target= sample['image'], sample['label']
            image, target= image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            main_out, aux_out = self.model(image)
            main_loss = self.criterion(main_out, target.long())
            aux_loss = self.criterion(aux_out, target.long())
            loss = main_loss +  0.4*aux_loss
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        for i, sample in enumerate(tbar):
            image, target = sample[0]['image'], sample[0]['label']
            image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                main_out, _ = self.model(image)
            pred = main_out.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target, pred)

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        ious = self.evaluator.Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        MDice, Dice = self.evaluator.Dice_Score()
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        self.writer.add_scalar('val/Dice', MDice, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc: {}, Acc_class: {}, mIoU: {}, fwIoU: {}, Mean Dice: {}".format(Acc, Acc_class, mIoU, FWIoU, MDice))
        print("Per-class Dice scores:")
        for class_idx, dice in Dice.items():
            print(f"Class {class_idx}: Dice = {dice:.4f}")
        print('IoUs: ', ious)

        if mIoU > self.best_pred:
            self.best_pred = mIoU
            self.save_checkpoint({
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, self.args.savepath + 'stage2_checkpoint_trained_on_'+self.args.dataset + '.pth')

    def load_the_best_checkpoint(self):
        checkpoint = torch.load(self.args.savepath + 'stage2_checkpoint_trained_on_' + self.args.dataset + '.pth')
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)

    def test(self, epoch, args):
        device = torch.device('cuda')
        self.model.to(device)
        if args.dataset == 'digestpath':
            palette = [239, 239, 239, 63, 125, 88]
        elif args.dataset == 'hph':
            palette = [239, 239, 239, 66, 27, 155]
        elif args.dataset == 'glas':
            palette = [239, 239, 239, 190, 61, 42]
        elif args.dataset == 'wien':
            palette = [239, 239, 239, 255, 250, 141]

        self.load_the_best_checkpoint()
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')

        for i, sample in enumerate(tbar):
            image, target = sample[0]['image'], sample[0]['label']
            png_name = sample[1][0]
            png_name = png_name.split('/')[-1][:-4]
            image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                main_out, _ = self.model(image)
            pred = main_out.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            save_dir = os.path.join(args.dataroot, 'test', 'pred_IBDR')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            visualimg = Image.fromarray(np.squeeze(pred).astype(np.uint8), "P")
            visualimg.putpalette(palette)
            visualimg.save(os.path.join(save_dir, png_name + '.png'), format='PNG')
            self.evaluator.add_batch(target, pred)

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        ious = self.evaluator.Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        MDice, Dice = self.evaluator.Dice_Score()
        MF1, F1s = self.evaluator.F1_Score()
        mAP, APs = self.evaluator.mAP()
        self.writer.add_scalar('val/F1', MF1, epoch)    
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        self.writer.add_scalar('val/Dice', MDice, epoch)
        self.writer.add_scalar('val/mAP', mAP, epoch)
        print('Test:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc: {}, Acc_class: {}, mIoU: {}, fwIoU: {}, Mean Dice: {}, mAP: {}".format(Acc, Acc_class, mIoU, FWIoU, MDice, mAP))
        print("Per-class Dice scores:")
        for class_idx, dice in Dice.items():
            print(f"Class {class_idx}: Dice = {dice:.4f}")
        print('IoUs: ', ious)
        print("Mean F1: {}".format(MF1))
        print("Per-class F1 scores:")
        for class_idx, f1 in F1s.items():
            print(f"Class {class_idx}: F1 = {f1:.4f}")
        print("Per-class AP scores:")
        for class_idx, ap in enumerate(APs):
            print(f"Class {class_idx}: AP = {ap:.4f}")

def main():
    parser = argparse.ArgumentParser(description="WSSS Stage2")
    parser.add_argument('--dataroot', type=str, default=r'./IBDR/datasets/HPH')
    parser.add_argument('--dataset', type=str, default='hph')
    parser.add_argument('--savepath', type=str, default='checkpoints/')
    parser.add_argument('--workers', type=int, default=8, metavar='N')
    parser.add_argument('--n_class', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR')
    parser.add_argument('--lr-scheduler', type=str, default='poly',choices=['poly', 'step', 'cos'])
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M')
    parser.add_argument('--nesterov', action='store_true', default=False )
    args = parser.parse_args()
    print(args)
    trainer = Trainer(args)
    for epoch in range(trainer.args.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)
    trainer.test(0, args)
    trainer.writer.close()

if __name__ == "__main__":
   main()
   