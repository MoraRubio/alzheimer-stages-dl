# Classification of Alzheimer's Disease stages from Magnetic Resonance Images using Deep Learning

## Authors

Alejandro Mora-Rubio<sup>1</sup>, Mario Alejandro Bravo-Ortiz<sup>1</sup>, Sebastián Quiñones-Arredondo<sup>1</sup>, Jose Manuel Saborit-Torres<sup>2</sup>, Gonzalo A. Ruz<sup>3,4,5</sup>, and Reinel Tabares-Soto<sup>1,3,6</sup>

[1] Department of Electronics and Automation, Universidad Autónoma de Manizales, Manizales 170001, Colombia

[2] Unidad Mixta de Imagen Biomédica FISABIO-CIPF, Fundación para el Fomento de la Investigación Sanitario y Biomédica de la Comunidad Valenciana, Valencia 46020, Spain

[3] Universidad Adolfo Ibáñez, Facultad de Ingeniería Ciencias, Santiago, 7941169, Chile

[4] Center of Applied Ecology and Sustainability (CAPES), Santiago, 8331150, Chile

[5] Data Observatory Foundation, Santiago, 7941169, Chile

[6] University of Caldas, 27985, Department of Systems and Informatics, Manizales, Colombia

## Abstract

Alzheimer's Disease (AD) is a progressive type of dementia characterized by loss of memory and other cognitive abilities, including speech. Since AD is a progressive disease, detection in the early stages is essential for the appropriate care of the patient across all stages, going from asymptomatic to a stage known as Mild Cognitive Impairment (MCI), and then progressing to dementia and severe dementia. Along with cognitive tests, evaluation of the brain morphology is the primary tool for AD diagnosis, where atrophy and loss of volume of the frontotemporal lobe are common features in patients who suffer from the disease. Regarding medical imaging techniques, Magnetic Resonance Imaging (MRI) scans are one of the methods used by specialists to assess brain morphology. Recently, with the rise of Deep Learning (DL) and its successful implementation in medical imaging applications, it is of growing interest in the research community to develop computer-aided diagnosis systems that can help physicians to detect this disease, especially in the early stages where macroscopic changes are not so easily identified. This paper presents a DL-based approach to classifying MRI scans in the different stages of AD, using a curated set of images from Alzheimer's Disease Neuroimaging Initiative (ADNI) and Open Access Series of Imaging Studies (OASIS) databases. Our methodology involves image preprocessing using FreeSurfer, spatial data-augmentation operations, such as rotation, flip, and random zoom during training, and state-of-the-art 3D Convolutional Neural Networks such as EfficientNet, DenseNet, and a custom siamese network. With this approach, the detection percentage of AD vs Control is around 85\%, Early MCI vs Control 67\%, and MCI vs Control 66\%.

## Materials and Methods

The implementation of DL models and training was done using [MONAI](https://docs.monai.io/en/stable) framework, it facilitates the development with the included architectures and preprocessing operations, as well as, allowing the use of custom models such as the [Siamese3D](models/siamese3D.py) used in this work.

The FreeSurfer software was utilized in the study and the command line used for processing the data is provided below:

`Recon-all -s $SUBJECT_NAME -i $INPUT_PATH -sd $PATH_RESULTS -all -cw256 -ba-labels`

The arguments included in the command line are explained as follows:

- `-s $SUBJECT_NAME`: specifies the ID of the different subjects.
- `-i $INPUT_PATH`: denotes the path where the images of the subjects are located.
- `-sd $PATH_RESULTS`: defines the path where the results will be saved.
- `-all`: applies all of the steps available in FreeSurfer.
- `-cw256`: reduces the size of the magnetic resonance imaging (MRI) image by cropping it to 256 pixels, which is done using the `mri_convert` command.
- `-ba-labels`: includes the volumes of the Brodmann areas in the analysis.

## Data

Data from [ADNI](https://adni.loni.usc.edu) and [OASIS](https://www.oasis-brains.org) can be accesed by request in their corresponding websites. The search parameters for ADNI database are presented in the [_IDASearch.pdf_](partition_tables/IDASearch.pdf) file; furthermore, only the 3 Tesla, T1 weighted, sagittal plane images with slice thickness between 1 and 1.5 mm were used. The _tsv_ files on the [partition_tables](partition_tables/) folder contain 10 different train, validation and test partitions maintaining a correct distribution of the subjects among the sets.

## Environment

Use anaconda and the provided YAML file to replicate the programming environment `conda env create -f pytorch_monai.yml`.
