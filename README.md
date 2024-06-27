# Many-to-many Bilateral Cross-modality Cluster Matching (MBCCM)
Pytorch Code of MULT method for Cross-Modality Person Re-Identification (Visible Thermal Re-ID) on RegDB dataset [1] and SYSU-MM01 dataset [2]. 

We adopt the two-stream network structure introduced in [3]. ResNet50 is adopted as the backbone.

*Both of these two datasets may have some fluctuation due to random spliting. The results might be better by finetuning the hyper-parameters. 

### 1. Prepare the datasets.

- (1) RegDB Dataset [1]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.

    - (Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1)" on their website). 

    - A private download link can be requested via sending me an email (mangye16@gmail.com). 
  
- (2) SYSU-MM01 Dataset [2]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

   - run `python pre_process_sysu.py` to pepare the dataset, the training data will be stored in ".npy" format.

### 2. Requirements
+ faiss_cpu==1.7.4
+ infomap==2.7.1
+ matplotlib==3.8.0
+ numpy==1.26.2
+ Pillow==10.0.1
+ Pillow==10.1.0
+ scikit_learn==1.3.1
+ scipy==1.11.4
+ seaborn==0.13.0
+ torch==2.0.1
+ torchvision==0.15.2
+ tqdm==4.66.1

### 3. Training.
  Train a model on SYSU-MM01:
  ```bash
python3 MULT_master/main.py --dataset sysu --batch-size 12 --num_pos 12 --eps 0.6 --pretrained False
```
  Train a model pretrained by the DCL [4] framework on SYSU-MM01:
  ```bash
python3 MULT_master/main.py --dataset sysu --batch-size 12 --num_pos 12 --eps 0.6 --pretrained True
```

Train a model on RegDB:
  ```bash
python3 MULT_master/tester.py --dataset regdb --batch-size 12 --num_pos 12 --eps 0.3 --pretrained False
```
Train a model pretrained by the DCL [4] framework on RegDB:
  ```bash
python3 MULT_master/tester.py --dataset regdb --batch-size 12 --num_pos 12 --eps 0.3 --pretrained True
```

You may need mannully define the data path first.

**Parameters**: More parameters can be found in the script.

### 4. Testing.

Test a model on SYSU-MM01: 
  ```bash
python tester.py --sysu_model_dir 'model_path'  --dataset sysu
```

Test a model on RegDB: 
  ```bash
python tester.py --sysu_model_dir 'model_path'  --dataset regdb
```

  - `--dataset`: which dataset "sysu" or "regdb".

[1] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.

[2] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.

[3] M. Ye, X. Lan, Z. Wang, and P. C. Yuen. Bi-directional Center-Constrained Top-Ranking for Visible Thermal Person Re-Identification. IEEE Transactions on Information Forensics and Security (TIFS), 2019.

[4] Bin Yang, Mang Ye, Jun Chen, and Zesen Wu. 2022. Augmented Dual-Contrastive Aggregation Learning for Unsupervised Visible-Infrared Person Re-Identification. In Proceedings of the 30th ACM International Conference on Multimedia (MM '22). Association for Computing Machinery, New York, NY, USA, 2843–2851. https://doi.org/10.1145/3503161.3548198

Contact: lfhe@stu.xidian.edu.cn