import torch
from data.data import MAEDataset, get_dataloader
from models.model_mae import MaskedAutoencoderViT
import os
from scripts.train import train_one_epoch

image_path = 'D:/user_data/individual_study/course_work/master-1fa/CS444/project/MAE/mydataset/images'
train_path = 'D:/user_data/individual_study/course_work/master-1fa/CS444/project/MAE/mydataset/annotations/trainval.txt'
test_path = 'D:/user_data/individual_study/course_work/master-1fa/CS444/project/MAE/mydataset/annotations/test.txt'

def save_model(model, file_name='weight/finetune.pth'):
    save_path = os.path.join(file_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")


class Args:
    def __init__(self):

        self.warmup_epochs = 5         
        self.lr = 1e-4                   
        self.min_lr = 1e-6                
        self.epochs = 100                
        self.mask_ratio = 0.75         
        self.accum_iter = 2             
        self.print_freq = 40
        self.epochs = 100     

if __name__ == '__main__':
    traindataset = MAEDataset(base_image_path = image_path, txt_path=train_path)
    testdataset = MAEDataset(base_image_path = image_path, txt_path=test_path)
    trainloader = get_dataloader(traindataset, batch_size=16)
    mae = MaskedAutoencoderViT()
    pretrain_weight_path = './weight/mae_pretrain_vit_base.pth'
    state_dict = torch.load(pretrain_weight_path)
    state_dict = state_dict['model']
    mae.load_state_dict(state_dict , strict=False)    
    # Create an instance of Args
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mae = mae.to(device)

    optimizer = torch.optim.AdamW(mae.parameters(), lr=args.lr, betas=(0.9, 0.95))

    for epoch in range(args.epochs):
        train_one_epoch(mae, trainloader, optimizer= optimizer,
                        device = device, epoch=epoch, log_writer=None,
                        args=args)
    save_model(mae)
    



