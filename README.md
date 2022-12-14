# Sum handwritten digit images@Machine Learning<br>
## Task: <br>
Design, implement, and evaluate a single neural network that outputs the sum of 2 hand written digit images as individual images.<br>
## A solution:
<br>The following is a sample program written in Pytorch(Python) that adds two digit images <br>
1. Generate new dataset from MNIST digits dataset by randomly pairing two digits. 
The paired digits has their sum as label
, i.e., we now have [0~18] numbers as labels for the new dataset<br>
   
   <img src = "./images/datasetPreview.png" width=600 height=400> <br>

      <img src="./images/addMnistDatasetDistribution.png" width=600 height=400>
      <br>the resulting dataset is imbalanced, as it can be seen from the above image <br>
   this samples imbalance will be taken care of during the training by considering per class sample weight<br>
   The class imbalance will as well influence the choice of evaluation metric.
2. Design a simple network for adding the two digit images
   <br>The designed architecture is derived from "Lenet" as it is has shown good classification results on MNIST dataset
     <img src="./images/modelArchitecture.png" width=600 height=400>
3. Train the Network
   <br>Dataset split into training and validation set[80% vs 20%]
   <br> Input to the network: image with two channels(each containg on digit) is fed to the network.
   1. this allows to apply the same filters to both images
   2. no distortion introduced to the digit images
   
   <img src = "./images/training.png" width=1000 height=400> <br>
   <br>The model converges around 100 epochs with ~95% accuracy
4. choose an evaluation metric
<br> 1. Accuracy per class
   <br> chosen as the newly generated dataset is imbalanced
   <br> Below is the confusion matrix on test dataset<br>
<img src = "./results/AddMnist_confusion_matrix.png" width=600 height=500> <br>
The following image depicts per Class accuracy. Although we have generated imbalanced data, pytorch provides a means of taking this imbalance into consideration.<br>
<img src = "./results/AddMnist_PerClassAccuracy.png" width=600 height=500> <br>
