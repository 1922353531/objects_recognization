# objects_recognition
## Abstract
This project is to implement objects recognition by CNN model. It contains three champion models in recent years. They are VGG-19, Inception-V4 and Inception-Resnet-V2, which is state-of-the-art in each series. You can choose one of them to train the network in your own datasets.
## How to get start
### Structure
There are three main py document, each of them contains a champion model.  
1.VGG-19.py contains the main model's structure of VGG-19.  
2.Inception-V4.py contains the main model's structure of Inception-V4.  
3.Inception-Resnet-V2.py contains the main model's structure of Inception-Resnet-V2.
### Training by yourself
After dealing with the datasets which you have download, just choose one model from three main py documents in model_training.py and run it is OK.
## PS
1.You can use any objects recognition datasets. If your datasets are too large, please use the fit_generator method to solve MemoryError.  
2.Each hyper-parameter in these models is setted according to their papers, which means you'd better not 8change them.
