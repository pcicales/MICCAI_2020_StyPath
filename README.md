# StyPath: Style-Transfer Data Augmentation For Robust Histology Image Classification

![](images/samples.png)

## Abstract
### The classification of Antibody Mediated Rejection (AMR) in kidney transplant remains challenging even for experienced nephropathologists; this is partly because histological tissue stain analysis is often characterized by low inter-observer agreement and poor reproducibility. One of the implicated causes for inter-observer disagreement is the variability of tissue stain quality between (and within) pathology labs, coupled with the gradual fading of archival sections. Variations in stain colors and intensities can make tissue evaluation difficult for pathologists, ultimately affecting their ability to describe relevant morphological features. Being able to accurately predict the AMR status based on kidney histology images is crucial for improving patient treatment and care. We propose a novel pipeline to build robust deep neural networks for AMR classification based on StyPath, a histological data augmentation technique that leverages a light weight style-transfer algorithm as a means to reduce sample-specific bias. Each image was generated in 1.84 ± 0.03 seconds using a single GTX TITAN V gpu and pytorch, making it faster than other popular histological data augmentation techniques. We evaluated our model using a Monte Carlo (MC) estimate of Bayesian performance and generate an epistemic measure of uncertainty to compare both the baseline and StyPath augmented models. We also generated Grad-CAM representations of the results which were assessed by an experienced nephropathologist; we used this qualitative analysis to elucidate on the assumptions being made by each model. Our results imply that our style-transfer augmentation technique improves histological classification performance (reducing error from 14.8% to 11.5%) and generalization ability.

## Results
![](images/results.png)

## Citation
### Please cite the following paper when using this code:
```
@inproceedings{cicalese2020stypath,
  title={StyPath: Style-Transfer Data Augmentation for Robust Histology Image Classification},
  author={Cicalese, Pietro Antonio and Mobiny, Aryan and Yuan, Pengyu and Becker, Jan and Mohan, Chandra and Van Nguyen, Hien},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={351--361},
  year={2020},
  organization={Springer}
}

```


