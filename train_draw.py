import torch
import torch.nn as nn
from torchvision import transforms , datasets
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model import DrawingDiffusion
from utils import ExponentialMovingAverage
import os
import math
import argparse
from PIL import Image

# from torchvision.datasets import MNIST


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # 디렉토리 내 모든 PNG 이미지 파일 경로 수집
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # 이미지 로드 및 변환 적용
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # 라벨은 없으므로 더미 값 0을 반환

def creat_dataloaders(batch_size, image_size=28, num_workers=4):
    # 이미지 전처리
    # preprocess = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5], [0.5])  # [0,1] 범위를 [-1,1]로 변환
    # ])
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # RGB 이미지를 그레이스케일로 변환
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # [0,1] 범위를 [-1,1]로 변환
    ])

    # 데이터 폴더의 기본 경로 및 카테고리 설정
    base_dir = os.path.join(os.path.dirname(__file__), "../sketch_data")
    categories = ["cat", "garden", "helicopter"]

    # train과 test 데이터셋 리스트 초기화
    train_datasets = []
    test_datasets = []

    # 각 카테고리의 train과 test 데이터를 CustomImageDataset으로 불러오기
    for category in categories:
        train_path = os.path.join(base_dir, category, "images_train")
        test_path = os.path.join(base_dir, category, "images_test")

        # CustomImageDataset으로 데이터를 불러오고 리스트에 추가
        train_datasets.append(CustomImageDataset(train_path, transform=preprocess))
        test_datasets.append(CustomImageDataset(test_path, transform=preprocess))

    # ConcatDataset으로 각 데이터셋 리스트를 하나로 합침
    train_dataset = ConcatDataset(train_datasets)
    test_dataset = ConcatDataset(test_datasets)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader

def parse_args():
    parser = argparse.ArgumentParser(description="Training DrawingDiffusion")
    parser.add_argument('--lr',type = float ,default=0.001)
    parser.add_argument('--batch_size',type = int ,default=128)    
    parser.add_argument('--epochs',type = int,default=100)
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=36)
    parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=64)
    parser.add_argument('--timesteps',type = int,help = 'sampling steps of DDPM',default=1000)
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--cpu',action='store_true',help = 'cpu training')

    args = parser.parse_args()

    return args


def main(args):
    device="cpu" if args.cpu else "cuda"
    train_dataloader,test_dataloader=creat_dataloaders(batch_size=args.batch_size,image_size=28)
    model=DrawingDiffusion(timesteps=args.timesteps,
                image_size=28,
                in_channels=1,
                base_dim=args.model_base_dim,
                dim_mults=[2,4]).to(device)

    #torchvision ema setting
    #https://github.com/pytorch/vision/blob/main/references/classification/train.py#L317
    adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer=AdamW(model.parameters(),lr=args.lr)
    scheduler=OneCycleLR(optimizer,args.lr,total_steps=args.epochs*len(train_dataloader),pct_start=0.25,anneal_strategy='cos')
    loss_fn=nn.MSELoss(reduction='mean')

    #load checkpoint
    if args.ckpt:
        ckpt=torch.load(args.ckpt)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

    global_steps=0
    for i in range(args.epochs):
        model.train()
        for j,(image,target) in enumerate(train_dataloader):
            noise=torch.randn_like(image).to(device)
            image=image.to(device)
            pred=model(image,noise)
            loss=loss_fn(pred,noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if global_steps%args.model_ema_steps==0:
                model_ema.update_parameters(model)
            global_steps+=1
            if j%args.log_freq==0:
                print("Epoch[{}/{}],Step[{}/{}],loss:{:.5f},lr:{:.5f}".format(i+1,args.epochs,j,len(train_dataloader),
                                                                    loss.detach().cpu().item(),scheduler.get_last_lr()[0]))
        ckpt={"model":model.state_dict(),
                "model_ema":model_ema.state_dict()}

        os.makedirs("results",exist_ok=True)
        torch.save(ckpt,"results/steps_{:0>8}.pt".format(global_steps))

        model_ema.eval()
        samples=model_ema.module.sampling(args.n_samples,clipped_reverse_diffusion=not args.no_clip,device=device)
        save_image(samples,"results/steps_{:0>8}.png".format(global_steps),nrow=int(math.sqrt(args.n_samples)))

if __name__=="__main__":
    args=parse_args()
    main(args)