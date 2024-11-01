# Cognitive Semantic Augmentation (CSA) LEO Satellite Networks for Earth Observation
This is a Pytorch implementation of CSA as proposed in the paper [Cognitive Semantic Augmentation (CSA) LEO Satellite Networks for Earth Observation](https://arxiv.org/abs/2410.21916)\
## Requirements
The codes are compatible with the packages:

- pytorch 1.8.0

- torchvision 0.9.0a0

- numpy 1.23.1

- tensorboardX 2.4

The code can be run on the datasets [EuroSAT](https://www.tensorflow.org/datasets/catalog/eurosat)
## Run experiments

### Training the CSA model 
 `python train2.py`
### Evaluating the trained CSA model
 `python3 evaluate_fl.py --mod apsk --dataset EuroSAT --latent_d 32 --save_root ./results-fl --name EuroSAT-num_e16-latent_d32-modapsk-psnr12.1-lam0.05`
 
## Block diagram of CSA LEO
![alt text](https://github.com/exhan100chou/CSA/blob/main/figs/2SatSEM.png)

## TXRX diagram of CSA
![alt text](https://github.com/exhan100chou/CSA/blob/main/figs/SEMtxrx.png)

## Top1 accuracy of confusion matrix using DTJSCC based on 16APSK Rician channel where PSNR=12dB and K=128.
![alt text](https://github.com/exhan100chou/CSA/blob/main/figs/cm_rician_12dB_128_apsk.png)

## Top1 accuracy of CSA satellite networks using DT-JSCC over 16APSK LEO Rician channel while DT-JSCC training at 4dB and CSA/federated learning training at 12dB
![alt text](https://github.com/exhan100chou/CSA/blob/main/figs/leo-top1-16apsk-semsat.png)

## Top1 accuracy comparison of CSA and non-CSA DT-JSCC K=32 systems while PSNR=12dB and 16APSK over LEO Rician channel.
![alt text](https://github.com/exhan100chou/CSA/blob/main/figs/CSAtable.JPG)

## Top1 accuracy using DT-JSCC based on 16APSK LEO Rician and LEO Rayleigh channel
![alt text](https://github.com/exhan100chou/CSA/blob/main/figs/leo-top1-16apsk.png)

## Citiation
```
@article{chou2024cognitivesemanticaugmentationleo,
      title={Cognitive Semantic Augmentation LEO Satellite Networks for Earth Observation}, 
      author={Hong-fu Chou and Vu Nguyen Ha and Prabhu Thiruvasagam and Thanh-Dung Le and Geoffrey Eappen and Ti Ti Nguyen and Duc Dung Tran and Luis M. Garces-Socarras and Juan Carlos Merlano-Duncan and Symeon Chatzinotas},
      year={2024},
      eprint={2410.21916},
      archivePrefix={arXiv},
      primaryClass={cs.NI},
      url={https://arxiv.org/abs/2410.21916}, 
}
```
