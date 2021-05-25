## PHIAF

Code and Datasets for "PHIAF: Prediction of phage-host interactions with GAN-based data augmentation and sequence-based feature fusion"

#### Developers

Menglu Li (mengluli@foxmail.com) and Wen Zhang (zhangwen@mail.hzau.edu.cn) from College of Informatics, Huazhong Agricultural University.

#### Datasets

- data/phage_dna_norm_features is a set of files with features encoded by DNA sequences corresponding to all phages.
- data/host_dna_norm_features is a set of files with features encoded by DNA sequences corresponding to all hosts.
- data/phage_protein_normfeatures is a set of files with features encoded by protein sequences corresponding to all phages.
- data/host_protein_normfeatures is a set of files with features encoded by protein sequences corresponding to all hosts.
- data/data_pos_neg.txt is the dataset used to train and test prediction model, which contain 312 positive and 312 negative samples (304 phages and 235 hosts).

#### Code

##### Environment Requirement

The code has been tested running under Python 3.7.9. The required packages are as follows:

- numpy == 1.19.1
- pandas == 1.1.3
- biopython == 1.78
- torch == 1.4.0+cpu
- keras == 2.3.1
- scikit-learn == 0.23.2
- tensorflow == 1.15.0

##### Usage

```
git clone https://github.com/mengluli-web/PHIAF
cd PHIAF/code
python generate_data.py   ####using GAN to generate pseudo positive samples
python main.py
```

Users can use their own data to train prediction models. First, using code/compute_dna_features.py and code/compute_protein_features.py to compute the features derived from DNA and protein sequences. 

In code/compute_dna_features.py, users need install iLearn tool [https://ilearn.erc.monash.edu/ or https://github.com/Superzchen/iLearn] and prepare .fasta file, this file is DNA sequences of all phages/hosts. 

In code/compute_protein_features.py, users need prepare .gb file of every phage/host, which download from NCBI database.  Then users using generate_data.py and main.py to predict PHI.

#### Contact

Please feel free to contact us if you need any help.
