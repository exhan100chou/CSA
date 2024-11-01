import os
from xmlrpc.client import Boolean 
import torch 
from tensorboardX import SummaryWriter
import torch.optim as optim 
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.utils import make_grid

from models.resnet import DTJSCC , DTJSCC_encode , DTJSCC_decode
from models.losses import RIBLoss, VAELoss 
from datasets.dataloader import get_data
from engine import train_one_epoch, test 
from utils.modulation import QAM, PSK, APSK

def main(args):
    """ Model and Opimizer """
    #model = DTJSCC(args.channels, args.latent_d, args.num_classes,num_embeddings=args.num_embeddings)
    model_encode = DTJSCC_encode(args.channels, args.latent_d,num_embeddings=args.num_embeddings)
    model_decode = DTJSCC_decode(args.latent_d, args.num_classes)

    model_encode.to(args.device if(torch.cuda.is_available()) else "cpu")
    model_decode.to(args.device if(torch.cuda.is_available()) else "cpu")
    
    optimizer_en = optim.AdamW(params=model_encode.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler_en = lr_scheduler.StepLR(optimizer_en, step_size=80,gamma=0.5)    
    optimizer_de = optim.AdamW(params=model_decode.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler_de = lr_scheduler.StepLR(optimizer_de, step_size=80,gamma=0.5)    
    """ Criterion """
    criterion = RIBLoss(args.lam)
    #criterion_aug = Cls_Loss(10)
    criterion.train()
    #criterion_aug.train()
    
    """ dataloader """
    dataloader_train =  get_data(args.dataset, args.N, n_worker= 8)
    dataloader_vali = get_data(args.dataset, args.N, n_worker= 8, train=False)
     
    """ writer """
    log_writer = SummaryWriter('./logs/'+ name)
    
    # fixed_images, _ = next(iter(dataloader_vali))
    # fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    # log_writer.add_image('original', fixed_grid, 0)
    
    current_epoch = 0
    best_acc = 0.0
    """ Some thing wrong here !!"""
    if os.path.isfile(path_to_backup):
        checkpoint = torch.load(path_to_backup, map_location='cpu')
        model_encode.load_state_dict(checkpoint['model_states'])
        optimizer_en.load_state_dict(checkpoint['optimizer_states'])
        model_decode.load_state_dict(checkpoint['model_states'])
        optimizer_de.load_state_dict(checkpoint['optimizer_states'])        
        current_epoch = checkpoint['epoch']  
        
    for epoch in range(current_epoch, args.epoches):
        if args.mod == 'qam':
            mod = QAM(args.num_embeddings, args.psnr)
        elif args.mod == 'psk':
            mod = PSK(args.num_embeddings, args.psnr)
        elif args.mod == 'apsk':
            mod = APSK(args.num_embeddings, args.psnr)    
        train_one_epoch(dataloader_train, model_encode, model_decode, optimizer_en=optimizer_en, optimizer_de=optimizer_de, criterion=criterion, writer=log_writer, epoch=epoch, mod=mod, args=args)
        scheduler_en.step()
        scheduler_de.step()
        if (epoch > 100): 
            #test_aug(dataloader_train,dataloader_vali, model, criterion=criterion_aug, writer=log_writer, epoch=epoch, mod=mod, args=args)
            acc1 = test(dataloader_vali, model_encode, model_decode, criterion=criterion, writer=log_writer, epoch=epoch, mod=mod, args=args)
            print('Epoch ', epoch)
            print('Best accuracy: ', best_acc)
        
            if (epoch == 0) or (acc1 > best_acc):
                best_acc = acc1
                with open('{0}/best_en.pt'.format(path_to_backup), 'wb') as f:
                    torch.save(
                    {
                    'epoch': epoch, 
                    'model_states': model_encode.state_dict(), 
                    'optimizer_states': optimizer_en.state_dict(),
                    }, f
                )
                with open('{0}/best_de.pt'.format(path_to_backup), 'wb') as f:
                    torch.save(
                    {
                    'epoch': epoch, 
                    'model_states': model_decode.state_dict(), 
                    'optimizer_states': optimizer_de.state_dict(),
                    }, f
                )    
        with open('{0}/model_en_{1}.pt'.format(path_to_backup, epoch + 1), 'wb') as f:
            torch.save(
            {
                'epoch': epoch, 
                'model_states': model_encode.state_dict(), 
                'optimizer_states': optimizer_en.state_dict(),
            }, f 
        )
        with open('{0}/model_de_{1}.pt'.format(path_to_backup, epoch + 1), 'wb') as f:
            torch.save(
            {
                'epoch': epoch, 
                'model_states': model_decode.state_dict(), 
                'optimizer_states': optimizer_de.state_dict(),
            }, f 
        )    


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    print('number of workers (default: {0})'.format(mp.cpu_count() - 1))

    parser = argparse.ArgumentParser(description='VDL')
    
    parser.add_argument('-d', '--dataset', type=str, default='CIFAR10', help='dataset name')
    parser.add_argument('-r', '--root', type=str, default='./trainded_models', help='The root of trained models')
    parser.add_argument('--device', type=str, default='cpu', help= 'The device')
    
    parser.add_argument('--mod', type=str, default='psk', help='The modulation')

    parser.add_argument('--num_latent', type=int, default=4, help='The number of latent variable')
    parser.add_argument('--latent_d', type=int, default=512, help='The dimension of latent vector')
    parser.add_argument('--channels', type=int, default=3, help='The channel')
    parser.add_argument('--num_classes', type=int, default=10, help='The number of the classes')
    
    parser.add_argument('-e', '--epoches', type=int, default=200, help='Number of epoches')
    parser.add_argument('--N', type=int, default=512, help='The batch size of training data')
    parser.add_argument('--lr', type=float, default=1e-3, help='learn rate')
    parser.add_argument('--maxnorm', type=float, default=1., help='The max norm of flip')
    
    parser.add_argument('--num_embeddings', type=int, default=16, help='The size of codebook')

    parser.add_argument('--lam', type=float, default=0.0, help='The lambda' )
    parser.add_argument('--psnr', type=float, default=8.0, help='The psnr' )

    parser.add_argument('--num_workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))

    args = parser.parse_args()
    args.n_iter = 0
    name = args.dataset + '-num_e'+ str(args.num_embeddings) + '-latent_d' + str(args.latent_d)+ '-mod'+ str(args.mod) + '-psnr'+ str(args.psnr)+ '-lam'+ str(args.lam)
 
    path_to_backup = os.path.join(args.root, name)
    if not os.path.exists(path_to_backup):
        print('Making ', path_to_backup, '...')
        os.makedirs(path_to_backup)

    if not os.path.exists('./logs'):
        print('Making logs...')
        os.makedirs('./logs')

    device = torch.device(args.device if(torch.cuda.is_available()) else "cpu")
    print('Device: ', device)

    main(args)
