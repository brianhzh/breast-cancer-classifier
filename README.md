## Breast Cancer Classifier

This project is a machine learning model built with PyTorch that classifies breast cancer images. 

The model was trained on the [BreakHis dataset](https://www.kaggle.com/datasets/ambarish/breakhis), which has two classifications: malignant or benign.

demo: https://brianhzh-breast-cancer-classifier-srcapp-blbdrs.streamlit.app/

To use, run

```
pip install -r requirements.txt
```

Sample training images:

<img width="850" height="290" alt="image" src="https://github.com/user-attachments/assets/fadb0692-31d1-4c96-8ee4-b642cd8d9ce2" />

---------------------------------------------

changelog:

---------------------------------------------

**first model** (adam optimizer, batch_size=32):

epochs: 5
train loss: ~0.15  
test loss: ~0.5  
issues: overfitting

**second model:** 

changes: 
- added dropout  
- data augmentation  
- OneCycleLR
- more layers
- optimizations

epochs: 30
train loss: ~0.24  
test loss: ~0.23  
test accuracy: 90.3%  

---------------------------------------------

Author: Brian Zhang


