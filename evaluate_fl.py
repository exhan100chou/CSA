import os
import torch
import torch.optim as optim 
import torch.optim.lr_scheduler as lr_scheduler
import copy
from tensorboardX import SummaryWriter
from datasets.dataloader import get_data, setup_clients, get_FL_data
from models.resnet import DTJSCC, DTJSCC_encode, DTJSCC_decode, server_aggregate_sgd,server_aggregate, server_aggregate_avg
from models.losses import RIBLoss, VAELoss 
from utils.modulation import QAM, PSK, APSK
from utils.accuracy import accuracy
from torchvision import models
from torchsummary import summary
from engine import train_one_epoch, test 
def client_update(model_encode, model_decode, dataloader_client, optimizer_en, optimizer_de, criterion, mod, args, local_epochs=1):
    """
    Function to update the client's model.
    Args:
    - model_encode: Local encoder model.
    - model_decode: Local decoder model.
    - dataloader_client: Client's data loader.
    - optimizer_en: Optimizer for the encoder.
    - optimizer_de: Optimizer for the decoder.
    - criterion: Loss function.
    - mod: Modulation function.
    - args: Arguments and configurations.
    - local_epochs: Number of epochs for local training.
    
    Returns:
    - Best model states for encoder and decoder after local training.
    - Best accuracy achieved on the client.
    - Pin to indicate if this client improved the accuracy (True if improved, False otherwise).
    """
    model_encode.train()
    model_decode.train()
    
    best_acc1 = 0.0
    best_model_encode_state = None
    best_model_decode_state = None
    accuracy_improved = False  # Pin to check if accuracy improved

    for epoch in range(local_epochs):
        for imgs, labs in dataloader_client:
            imgs = imgs.to(args.device if torch.cuda.is_available() else "cpu")
            labs = labs.to(args.device if torch.cuda.is_available() else "cpu")
            
            # Forward pass
            en_X, former_shape, dist = model_encode(imgs, mod=mod)
            features = model_decode(en_X, former_shape)
            outs = model_decode.head(features)
            
            loss, _, _ = criterion(dist, outs, labs)
            
            # Backpropagation
            optimizer_en.zero_grad()
            optimizer_de.zero_grad()
            loss.backward()
            if args.maxnorm > 0:
               torch.nn.utils.clip_grad_norm_(model_encode.parameters(), args.maxnorm)
               torch.nn.utils.clip_grad_norm_(model_decode.parameters(), args.maxnorm)            
            optimizer_en.step()
            optimizer_de.step()

        # Evaluate model on client data
        acc1, _, _ = eval_test(dataloader_client, model_encode, model_decode, mod, args)
        
        # If accuracy improves, update the model states and set the pin to True
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_model_encode_state = copy.deepcopy(model_encode.state_dict())
            best_model_decode_state = copy.deepcopy(model_decode.state_dict())
            accuracy_improved = True  # Pin is set to True if accuracy improved

    return best_model_encode_state, best_model_decode_state, best_acc1, accuracy_improved
def federated_learning_sgd(global_model_encode, global_model_decode, N, num_rounds, local_epochs, lr, criterion, log_writer, mod, args, data_set, batch_size):
    """
    Main function for federated learning using client-side AdamW and server-side SGD.
    
    Args:
    - global_model_encode: Global encoder model.
    - global_model_decode: Global decoder model.
    - N: Number of clients.
    - num_rounds: Number of communication rounds.
    - local_epochs: Number of local epochs per client.
    - lr: Learning rate for both client and server.
    - criterion: Loss function.
    - log_writer: TensorBoard writer for logging.
    - mod: Modulation object.
    - args: Arguments passed to the script.
    - data_set: Dataset to be used.
    - batch_size: Batch size for the DataLoader.
    
    Returns:
    - global_model_encode: Updated global encoder model.
    - global_model_decode: Updated global decoder model.
    """

    # Set up clients' data loaders
    clients = setup_clients(N, data_set, batch_size)
    val_loader = get_data(args.dataset, args.N, n_worker=8, train=False)

    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}/{num_rounds}")
        
        # Store gradients and accuracy from clients
        client_gradients_encode = []
        client_gradients_decode = []
        client_accs = []

        # Train each client locally
        for client_data in clients:
            local_model_encode = copy.deepcopy(global_model_encode)
            local_model_decode = copy.deepcopy(global_model_decode)

            # Use AdamW for client-side optimization
            optimizer_en = torch.optim.AdamW(local_model_encode.parameters(), lr=lr)
            optimizer_de = torch.optim.AdamW(local_model_decode.parameters(), lr=lr)

            # Perform local training and obtain gradients
            best_model_encode_state, best_model_decode_state, best_acc1, encode_grads, decode_grads = client_update_sgd(
                local_model_encode, local_model_decode, client_data, optimizer_en, optimizer_de, criterion, mod, args, local_epochs
            )

            client_accs.append(best_acc1)
            client_gradients_encode.append(encode_grads)
            client_gradients_decode.append(decode_grads)

        # Aggregate client gradients using server-side SGD
        global_model_encode = server_aggregate_sgd(global_model_encode, client_gradients_encode, lr)
        global_model_decode = server_aggregate_sgd(global_model_decode, client_gradients_decode, lr)

        # Validate the global model on the server-side data
        val_acc1, val_acc3, _ = eval_test(val_loader, global_model_encode, global_model_decode, mod, args)
        print(f"Validation - Round {round_num + 1}: Top-1 Accuracy: {val_acc1:.4f}, Top-3 Accuracy: {val_acc3:.4f}")

        # Log results
        log_writer.add_scalar('val/acc1', val_acc1, round_num)
        log_writer.add_scalar('val/acc3', val_acc3, round_num)

    return global_model_encode, global_model_decode



def federated_learning(global_model_encode, global_model_decode, N, num_rounds, local_epochs, lr, criterion, log_writer, mod, args, data_set, batch_size):
    """
    Main function for federated learning with best client model aggregation.
    Args:
    - global_model_encode: Global encoder model.
    - global_model_decode: Global decoder model.
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
    val_loader = get_FL_data(args.dataset,  args.N, n_worker=8, train=False)
    client_models = []
    client_accs = []    
    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}/{num_rounds}")


        # Train on each client
        for client_data in clients:
            local_model_encode = copy.deepcopy(global_model_encode)
            local_model_decode = copy.deepcopy(global_model_decode)

            optimizer_en = torch.optim.AdamW(local_model_encode.parameters(), lr=lr)
            optimizer_de = torch.optim.AdamW(local_model_decode.parameters(), lr=lr)

            # Get the best model and accuracy from the client, and also get the pin
            best_model_encode_state, best_model_decode_state, best_acc1, accuracy_improved = client_update(
                local_model_encode, local_model_decode, client_data, optimizer_en, optimizer_de, criterion, mod, args, local_epochs
            )

            # Append only if the accuracy improved (i.e., pin is True)
            if accuracy_improved:
               client_models.append((best_model_encode_state, best_model_decode_state))
               client_accs.append(best_acc1)

        # Aggregate best client models at the server
        global_model_encode, global_model_decode = server_aggregate_avg(global_model_encode, global_model_decode, client_models, client_accs)

        # Validation on the server-side
        val_acc1, val_acc3, _ = eval_test(val_loader, global_model_encode, global_model_decode, mod, args)
        print(f"Validation - Round {round_num + 1}: Top-1 Accuracy: {val_acc1:.4f}, Top-3 Accuracy: {val_acc3:.4f}")

        log_writer.add_scalar('val/acc1', val_acc1, round_num)
        log_writer.add_scalar('val/acc3', val_acc3, round_num)

    return global_model_encode, global_model_decode



def eval_test(datal, model_encode, model_decode, mod, args):
    acc1, acc3 = 0., 0.
    with torch.no_grad():
        model_encode.eval()
        model_decode.eval()
        for imgs, labs in datal:
            imgs = imgs.to(args.device if(torch.cuda.is_available()) else "cpu")
            labs = labs.to(args.device if(torch.cuda.is_available()) else "cpu")
            
            en_X  , former_shape, dist = model_encode(imgs, mod=mod)
            features = model_decode(en_X , former_shape)
            outs = model_decode.head(features) 
            acc = accuracy(outs, labs, (1,3))
            acc1 += acc[0].item()
            acc3 += acc[1].item()
    #print('Done!')
    
    return acc1/len(datal), acc3/len(datal), dist

def main(args):
    # model and dataloader
    if args.dataset == 'EuroSAT':
       model_encode = DTJSCC_encode(args.channels, args.latent_d,num_embeddings=args.num_embeddings)
       model_decode = DTJSCC_decode(args.latent_d, args.num_classes)   
    else:
        assert 1 == 0, args.dataset
    
    """ Criterion """
    criterion = RIBLoss(args.lam)
    #criterion_aug = Cls_Loss(10)
    criterion.train()
    
    log_writer = SummaryWriter('./logs/'+ name)    
    # Federate learning client dataloading
    dataloader_FLvali = get_data(args.dataset,  args.N, n_worker=8, train=False)

    # load global model for Federate learning 
    checkpoint_en = torch.load('{0}/best_en_fl.pt'.format(path_to_backup), map_location='cpu')
    model_encode.load_state_dict(checkpoint_en['model_states'], strict=False)
    model_encode.to(args.device)    
    checkpoint_de = torch.load('{0}/best_de_fl.pt'.format(path_to_backup), map_location='cpu')
    model_decode.load_state_dict(checkpoint_de['model_states'], strict=False)
    model_decode.to(args.device)      
    """ Start FL """
    optimizer_en = optim.AdamW(params=model_encode.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler_en = lr_scheduler.StepLR(optimizer_en, step_size=80,gamma=0.5)    
    optimizer_de = optim.AdamW(params=model_decode.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler_de = lr_scheduler.StepLR(optimizer_de, step_size=80,gamma=0.5)  
 
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
    model_encode, model_decode=federated_learning(model_encode, model_decode, N, num_rounds, local_epochs, args.lr, criterion, log_writer, mod, args, args.dataset, args.N)
    with open('{0}/best_en_fl.pt'.format(path_to_backup), 'wb') as f:
            torch.save(
                    {
                    'model_states': model_encode.state_dict(), 
                    'optimizer_states': optimizer_en.state_dict(),
                    }, f
                )
    with open('{0}/best_de_fl.pt'.format(path_to_backup), 'wb') as f:
                    torch.save(
                    {
                    'model_states': model_decode.state_dict(), 
                    'optimizer_states': optimizer_de.state_dict(),
                    }, f
                )    
    """ End FL """
    #Testing
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
        a1, a3, dist = eval_test(dataloader_FLvali,  model_encode, model_decode, mod, args)
        acc1s.append(a1)
        acc3s.append(a3)
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
    parser.add_argument('--num_clients', type=int, default=10, help='The number of Sats')
    parser.add_argument('--local_epochs', type=int, default=4, help='Number of local epoches')
    parser.add_argument('--num_fl_rounds', type=int, default=20, help='The round number of federated learning')  
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
    
    
        

    
        
    
    
    

