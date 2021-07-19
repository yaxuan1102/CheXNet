# CheXNet
## Binary Classification of Pneumonia


### Data
A total of 5863 chest radiography images in JPEG format.

|              | Normal | Pnemonia |
|--------------|--------|----------|
|     Train    |  1341  |   3875   |
|     Test     |  234   |    390   |
|  Validation  |  8     |    8     |


### Preprocess
#### 1. Create label list
- Norma: 0 1
- Pneumonia: 1 0

#### 2. Image preprocess
- Z-normalization
- Data augmentation: random horizontal flipping
- Resize: 256x256
- Crop: 224x224


### Model
- Source: https://github.com/zoogzog/chexnet
- Framework: PyTorch
- Model: DenseNet121
- Paramenters and settings:
|     Initial weight        |  Pretrained from ImageNet Dataset       |
|      Loss function        |   Binary cross entropy                  |
|        Optimizer          |  Adam (beta_1 = 0.9, beta_2 = 0.999)    |
|  Initial learning rate    |  10e-4                                  |
|     Batch size            |  16                                     |


### Evaluation Matrices
- ROC
- AUC
- Confusion matrix



### Training Performance
- AUC: ~0.999 (for averaged AUC and both categories)
- Accuracy: 0.9967
- Sensitivity: 0.988
- Specificity: 0.9997

### Validation Performance
- AUC: 1.0 (for averaged AUC and both categories)
- Accuracy: 0.8125
- Sensitivity: 1.0
- Specificity: 0.6154


### Testing Results
- AUC: ~0.995 (for averaged AUC and both categories)
- Accuracy: 0.9615
- Sensitivity: 0.9907
- Specificity: 0.9463
