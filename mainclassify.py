import torch
from data.data import Classify_Dataset, get_dataloader
from models.model_cls import MaskedAutoencoderClassify
from models.model_linearprob import MaskedAutoencoderLinearProbe
import os

image_path = 'D:/user_data/individual_study/course_work/master-1fa/CS444/project/MAE/mydataset/images'
train_path = 'D:/user_data/individual_study/course_work/master-1fa/CS444/project/MAE/mydataset/annotations/trainval.txt'
test_path = 'D:/user_data/individual_study/course_work/master-1fa/CS444/project/MAE/mydataset/annotations/test.txt'

class Args:
    def __init__(self):

        self.warmup_epochs = 5         
        self.lr = 1e-4                   
        self.min_lr = 1e-6                
        self.epochs = 100                
        self.mask_ratio = 0.75         
        self.accum_iter = 2             
        self.print_freq = 40
        self.epochs = 20    

if __name__ == '__main__':
    traindataset = Classify_Dataset(base_image_path = image_path, txt_path=train_path)
    testdataset = Classify_Dataset(base_image_path = image_path, txt_path=test_path)
    trainloader = get_dataloader(traindataset, batch_size=16)
    testloader = get_dataloader(testdataset, batch_size=16)
    pretrain_weight_path = './weight/finetune.pth'
    state_dict = torch.load(pretrain_weight_path)
    model = MaskedAutoencoderClassify()
    model.load_state_dict(state_dict , strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    args = Args()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (imgs, classid) in enumerate(trainloader):
            imgs, classid = imgs.to(device), classid.to(device)
            optimizer.zero_grad()
            loss, logits = model(imgs, classid)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(logits, 1)
            total += classid.size(0)
            correct += (predicted == classid).sum().item()

            if (i + 1) % args.print_freq == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(trainloader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = (correct / total) * 100
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        if epoch >= args.warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(args.min_lr, param_group['lr'] * 0.9)
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, classid in testloader:
            imgs, classid = imgs.to(device), classid.to(device)
            predicted = model.predict(imgs)
            total += classid.size(0)
            correct += (predicted == classid).sum().item()

    