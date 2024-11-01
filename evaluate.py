import os
import torch
import torch.optim as optim 
import torch.optim.lr_scheduler as lr_scheduler
import copy
from tensorboardX import SummaryWriter
from datasets.dataloader import get_data, setup_clients, get_FL_data
from models.resnet import DTJSCC, server_aggregate_sgd,server_aggregate_avg1
from models.losses import RIBLoss, VAELoss 
from utils.modulation import QAM, PSK, APSK
from utils.accuracy import accuracy
from torchvision import models
from torchsummary import summary
from engine import train_one_epoch, test 
def client_update(model, optimizer, dataloader_client, criterion, mod, args, local_epochs=1):
    """
    Function to update the client's model.
    Args:
    - model: Local model.
    - optimizer: Optimizer for the model.
    - dataloader_client: Client's data loader.
    - criterion: Loss function.
    - mod: Modulation function.
    - args: Arguments and configurations.
    - local_epochs: Number of epochs for local training.
    
    Returns:
    - Updated model weights (state dictionary) after local training.
    """
    model.train()  # Set the model to training mode
    best_acc1 = 0.0
    best_model_state = None

    for epoch in range(local_epochs):
        for imgs, labs in dataloader_client:
            imgs = imgs.to(args.device if torch.cuda.is_available() else "cpu")
            labs = labs.to(args.device if torch.cuda.is_available() else "cpu")
            
            # Forward pass
            outs, dist = model(imgs, mod=mod)
            
            loss, _, _ = criterion(dist, outs, labs)  # Compute loss
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            if args.maxnorm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.maxnorm)  # Apply gradient clipping
            optimizer.step()  # Update model weights
        # Evaluate model on client data
        acc1, _, _ = eval_test(dataloader_client, model, mod, args)
        
        # If accuracy improves, update the model states and set the pin to True
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_model_state  = copy.deepcopy(model.state_dict())
            
    # Return the updated model weights (state dictionary)
    return best_model_state, best_acc1


def federated_learning(global_model, N, num_rounds, local_epochs, lr, criterion, log_writer, mod, args, data_set, batch_size):
    """
    Main function for federated learning with best client model aggregation.
    Args:
    - global_model: Global model.
    - N: Number of clients.
    - num_rounds: Number of communication rounds.
    - local_epochs: Number of local epochs per client.
    - lr: Learning rate.
    - criterion: Loss function.
    - log_writer: Summary writer for logging.
    - mod: Modulation object.
    - args: Arguments.
    - data_set: Dataset name.
    - batch_size: Batch size.
    """

    # Setup clients and server validation data
    clients = setup_clients(N, data_set, batch_size)
    val_loader = get_FL_data(args.dataset, args.N, n_worker=8, train=False)
    client_models = []
    client_accs = []     
    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}/{num_rounds}")

        # Train on each client
        for client_data in clients:
            local_model = copy.deepcopy(global_model)

            optimizer = torch.optim.AdamW(local_model.parameters(), lr=lr)

            # Get the updated model weights from the client after local training
            client_model_weights , best_acc1= client_update(
                local_model, optimizer, client_data, criterion, mod, args, local_epochs
            )

            # Append the client model's weights (for aggregation)
            client_models.append(client_model_weights)
            client_accs.append(best_acc1)
        # Aggregate the client models at the server using SGD
        global_model = server_aggregate_avg1(global_model, client_models, client_accs)

        # Validation on the server-side
        val_acc1, val_acc3, _ = eval_test(val_loader, global_model, mod, args)
        print(f"Validation - Round {round_num + 1}: Top-1 Accuracy: {val_acc1:.4f}, Top-3 Accuracy: {val_acc3:.4f}")

        log_writer.add_scalar('val/acc1', val_acc1, round_num)
        log_writer.add_scalar('val/acc3', val_acc3, round_num)

    return global_model




def eval_test(datal, model, mod, args):
    acc1, acc3 = 0., 0.
    with torch.no_grad():
        model.eval()
        for imgs, labs in datal:
            imgs = imgs.to(args.device if(torch.cuda.is_available()) else "cpu")
            labs = labs.to(args.device if(torch.cuda.is_available()) else "cpu")
            
            outs, dist = model(imgs, mod=mod)
            acc = accuracy(outs, labs, (1,3))
            acc1 += acc[0].item()
            acc3 += acc[1].item()
    
    
    return acc1/len(datal), acc3/len(datal), dist

def main(args):
    # model and dataloader
    if args.dataset == 'EuroSAT':
       model = DTJSCC(args.channels, args.latent_d,args.num_classes,num_embeddings=args.num_embeddings)
    else:
        assert 1 == 0, args.dataset
    dataloader_vali = get_data(args.dataset, 256, n_worker=8, train=False)
   # load global model for Federate learning 
    checkpoint = torch.load('{0}/best.pt'.format(path_to_backup), map_location='cpu')
    model.load_state_dict(checkpoint['model_states'], strict=False)
    model.to(args.device)      
    """ Start FL """
    criterion = RIBLoss(args.lam)
    #criterion_aug = Cls_Loss(10)
    criterion.train()
    log_writer = SummaryWriter('./logs/'+ name)    


    optimizer= optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=80,gamma=0.5)  
  
    # Federated learning parameters
    N = args.num_clients  # Number of clients
    num_rounds = args.num_fl_rounds # Number of federated learning rounds
    local_epochs = args.local_epochs   # Number of epochs per client
    print(f"Number of clients {N}")
    # Define modulation
    if args.mod == 'qam':
        mod = QAM(args.num_embeddings, args.psnr)
    elif args.mod == 'psk':
        mod = PSK(args.num_embeddings, args.psnr)
    elif args.mod == 'apsk':
        mod = APSK(args.num_embeddings, args.psnr)
    else:
        raise NotImplementedError(f"Modulation {args.mod} not supported")

    # Run federated learning
    model=federated_learning(model, N, num_rounds, local_epochs, args.lr, criterion, log_writer, mod, args, args.dataset, args.N)
    with open('{0}/best_fl.pt'.format(path_to_backup), 'wb') as f:
        torch.save(
                 {
                  'model_states': model.state_dict(), 
                  'optimizer_states': optimizer.state_dict(),
                 }, f
              )
    """ End FL """
    PSNRs = list(range(2, 21, 2))
    acc1s = []
    acc3s = []
    dist_re = None
    for psnr in PSNRs:
        if args.mod == 'qam':
            mod = QAM(args.num_embeddings, psnr)
        elif args.mod == 'psk':
            mod = PSK(args.num_embeddings, psnr)
        elif args.mod == 'apsk':
            mod = APSK(args.num_embeddings, psnr)    
        a1, a3, dist = eval_test(dataloader_vali, model, mod, args)
        acc1s.append(a1)
        acc3s.append(a3)
        #print(f"Validation - Round { psnr}: Top-1 Accuracy: {a1:.4f}, Top-3 Accuracy: {a3:.4f}")
        dist_re = dist
    with open('{0}.pt'.format(path_to_save), 'wb') as f:
        torch.save(
                {
                    'acc1s':acc1s,
                    'acc3s':acc3s,
                    'dist':dist_re
                }, f
            )
        
if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(description='ways')
    parser.add_argument('--mod', type=str, default='apsk', help='The modulation')
    parser.add_argument('--num_latent', type=int, default=4, help='The number of latent variable')
    parser.add_argument('--latent_d', type=int, default=512, help='The dimension of latent vector')
    parser.add_argument('--channels', type=int, default=3, help='The channel')
    parser.add_argument('-d', '--dataset', type=str, default='CIFAR10', help='dataset name')
    parser.add_argument('-r', '--root', type=str, default='./trainded_models', help='The root of trained models')
    parser.add_argument('--device', type=str, default='cpu', help= 'The device')
    parser.add_argument('--num_embeddings', type=int, default=16, help='The size of codebook')
    parser.add_argument('--num_classes', type=int, default=10, help='The number of the classes')
    parser.add_argument('--name', type=str, default='ta', help= 'The trained model')
    parser.add_argument('--save_root', type=str, default='./results-fl', help= 'The root of result')
    parser.add_argument('--lam', type=float, default=0.05, help='The lambda' )
    parser.add_argument('--psnr', type=float, default=12.0, help='The psnr' )
    parser.add_argument('-e', '--epoches', type=int, default=200, help='Number of epoches')
    parser.add_argument('--N', type=int, default=512, help='The batch size of training data')
    parser.add_argument('--lr', type=float, default=1e-3, help='learn rate') 
    parser.add_argument('--maxnorm', type=float, default=1., help='The max norm of flip')  
    parser.add_argument('--num_clients', type=int, default=2, help='Number of Sats')
    parser.add_argument('--local_epochs', type=int, default=4, help='Number of local epoches')
    parser.add_argument('--num_fl_rounds', type=int, default=100, help='The round number of federated learning')  
    args = parser.parse_args()
    path_to_backup = os.path.join(args.root, args.name)

    if not os.path.exists(args.save_root):
        print('making results...')
        os.makedirs(args.save_root)
    name = args.dataset + '-num_e'+ str(args.num_embeddings) + '-latent_d' + str(args.latent_d)+ '-mod'+ str(args.mod) + '-psnr'+ str(args.psnr)+ '-lam'+ str(args.lam)      
    name_fl = args.dataset + '-num_e'+ str(args.num_embeddings) + '-latent_d' + str(args.latent_d)+ '-mod'+ str(args.mod) + '-psnr'+ str(args.psnr)+ '-lam'+ str(args.lam)  +'-client_n' + str(args.num_clients)   +'-fl_n' + str(args.num_fl_rounds) 
    path_to_save = os.path.join(args.save_root, name_fl)
    args.n_iter = 0    
    device = torch.device(args.device if(torch.cuda.is_available()) else "cpu")
    print('Device: ', device)
    
    main(args)
    
    
        

    
        
    
    
    

