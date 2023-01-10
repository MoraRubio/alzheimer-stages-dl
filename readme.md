# Classification of Alzheimer's Disease stages from Magnetic Resonance Images using Deep Learning
## Authors


## Abstract 
Alzheimer's Disease (AD) is a progressive type of dementia characterized by loss of memory and other cognitive abilities, including speech. Since AD is a progressive disease, detection in the early stages is essential for the appropriate care of the patient across all stages, going from asymptomatic to a stage known as Mild Cognitive Impairment (MCI), and then progressing to dementia and severe dementia. Along with cognitive tests, evaluation of the brain morphology is the primary tool for AD diagnosis, where atrophy and loss of volume of the frontotemporal lobe are common features in patients who suffer from the disease. Regarding medical imaging techniques, Magnetic Resonance Imaging (MRI) scans are one of the methods used by specialists to assess brain morphology. Recently, with the rise of Deep Learning (DL) and its successful implementation in medical imaging applications, it is of growing interest in the research community to develop computer-aided diagnosis systems that can help physicians to detect this disease, especially in the early stages where macroscopic changes are not so easily identified. This paper presents a DL-based approach to classifying MRI scans in the different stages of AD, using a curated set of images from Alzheimer's Disease Neuroimaging Initiative (ADNI) and Open Access Series of Imaging Studies (OASIS) databases. Our methodology involves image preprocessing using FreeSurfer, spatial data-augmentation operations, such as rotation, flip, and random zoom during training, and state-of-the-art 3D Convolutional Neural Networks such as EfficientNet, DenseNet, and a custom siamese network. With this approach, the detection percentage of AD vs Control is around 85\%, Early MCI vs Control 67\%, and MCI vs Control 66\%.

## Materials and Methods
The implementation of DL models and training was done using [MONAI](https://docs.monai.io/en/stable) framework, it facilitates the development with the included architectures and preprocessing operations, as well as, allowing the use of custom models such as the [Bilinear](models/bilinear3D.py) used in this work.

## Environment
Use anaconda and the provided YAML file to replicate the programming environment `conda env create -f pytorch_monai.yml`.