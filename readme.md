# Training a Pizza Slice Angle Detector


```python
## Define imports

## Packages/environment contained in "requirements.yml"

## See 'dataloader.py' for custom dataloader class
from dataloader import create_data, PizzaDataset
## See 'model.py' for custom model class
from model import KeypointDetector
## See 'image_transforms.py' for custom image transforms
from image_transforms import *


## Some generic functions
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import v2

```

## Load Data from SQL Database and Create Train-Test Split


```python
dl = create_data('./data/pizza_database.db','pizza_table')
train, valid = dl.split_data(0.8, ['index','data_path','x1','y1','x2','y2','other'])
```

## Define Image Transform Pipeline

Uses a combination of custom functions and those from the Torch library.

The rescale function also rescales the label coordinates.

NB: Due to time constraints, I decided to just normalize the image dimensions of each image. In some instances the aspect ratios are altered, which potentially causes a change in the angle of the pizza slice. 

Inference can be run on the normalized data, and there is a function in the dataloader class ('transform_pred_to_normal') which rescales the keypoints back to the original height and width of the image before pre-processing.


```python
transforms = v2.Compose([
    Rescale((224, 224)),
    Normalize(),
    v2.ToDtype(torch.float32),
])
```

## Transform Train Dataset

Create both transformed and untransformed datasets.

**Moving forward: x1,y1 is the tip and taken to be the origin point when calculating the angle relative to the vertical i.e. Y-axis**

The intention of both the data loader and the rest of this pipeline allows for **more training data to be added to the SQL database** so the model can be improved. 

Given more training data, batch size could be increased also. 


```python
untransformed_train_dataset = PizzaDataset(train, './data/images')
transformed_train_dataset = PizzaDataset(train, './data/images', transforms)

print(f'Size of Training Dataset: {len(train)}')

train
```

    Size of Training Dataset: 4





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>data_path</th>
      <th>x1</th>
      <th>y1</th>
      <th>x2</th>
      <th>y2</th>
      <th>other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>676378</td>
      <td>pizza0.jpg</td>
      <td>65</td>
      <td>905</td>
      <td>108</td>
      <td>875</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>603024</td>
      <td>pizza1.jpg</td>
      <td>99</td>
      <td>243</td>
      <td>99</td>
      <td>224</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>893981</td>
      <td>pizza3.jpg</td>
      <td>1547</td>
      <td>430</td>
      <td>1466</td>
      <td>420</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>162132</td>
      <td>pizza4.jpg</td>
      <td>270</td>
      <td>409</td>
      <td>325</td>
      <td>399</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



## Visualize Examples

NB: Note Y-axis is flipped when interpreting printed angle. 


```python
## Untransformed Image

eg0 = transformed_train_dataset.__getitem__(0)
eg0_u = untransformed_train_dataset.__getitem__(0)

dl.visualize_matrix_with_coordinates(eg0_u['image'], eg0_u['keypoints'], flip_y=False)

print(f"Angle of slice: {dl.calculate_clockwise_angle(eg0['keypoints'])}")
```


    
![png](pizza_walkthrough_files/pizza_walkthrough_9_0.png)
    


    Angle of slice: 132.92996934695887



```python
## Transformed Image

dl.visualize_matrix_with_coordinates(eg0['image'],eg0['keypoints'], flip_y=False)

print(f"Angle of slice: {dl.calculate_clockwise_angle(eg0_u['keypoints'])}")
```


    
![png](pizza_walkthrough_files/pizza_walkthrough_10_0.png)
    


    Angle of slice: 124.90249561592474



```python
## Untransformed Image

eg1 = transformed_train_dataset.__getitem__(1)
eg1_u = untransformed_train_dataset.__getitem__(1)

dl.visualize_matrix_with_coordinates(eg1_u['image'],eg1_u['keypoints'], flip_y=False)
print(f"Angle of slice: {dl.calculate_clockwise_angle(eg1['keypoints'])}")
```


    
![png](pizza_walkthrough_files/pizza_walkthrough_11_0.png)
    


    Angle of slice: 180.0


    /home/sandippanesar/Desktop/pizza_angle_prediction/dataloader.py:179: RuntimeWarning: divide by zero encountered in double_scalars
      m = (points[3] - points[2]) / (points[1] - points[0])



```python
## Transformed Image

dl.visualize_matrix_with_coordinates(eg1['image'],eg1['keypoints'], flip_y=False)
print(f"Angle of slice: {dl.calculate_clockwise_angle(eg1_u['keypoints'])}")
```


    
![png](pizza_walkthrough_files/pizza_walkthrough_12_0.png)
    


    Angle of slice: 180.0



```python
## Untransformed Image

eg2 = transformed_train_dataset.__getitem__(2)
eg2_u = untransformed_train_dataset.__getitem__(2)

dl.visualize_matrix_with_coordinates(eg2_u['image'],eg2_u['keypoints'], flip_y=False)
print(f"Angle of slice: {dl.calculate_clockwise_angle(eg2['keypoints'])}")
```


    
![png](pizza_walkthrough_files/pizza_walkthrough_13_0.png)
    


    Angle of slice: 80.6524221903351



```python
## Transformed Image

dl.visualize_matrix_with_coordinates(eg2['image'],eg2['keypoints'], flip_y=False)
print(f"Angle of slice: {dl.calculate_clockwise_angle(eg2['keypoints'])}")
```


    
![png](pizza_walkthrough_files/pizza_walkthrough_14_0.png)
    


    Angle of slice: 80.6524221903351


## Create DataLoader Class For Datasets


```python
train_loader = DataLoader(transformed_train_dataset, batch_size=1, shuffle=True)
untransformed_test_dataset = PizzaDataset(valid, './data/images')
transformed_test_dataset = PizzaDataset(valid, './data/images', transforms)
test_loader = DataLoader(transformed_test_dataset, batch_size=1, shuffle=True)
```

## Define Model Criteria For Training

L1Loss chosen specifically for keypoint detection task, over MSE or other loss functions. 

Has optional dropout layer.

Model 'KeypointDetector' contains: <br>
    - A convolutional layer <br>
    - A dropout layer <br>
    - A max pooling layer <br>
    - Another convolutional layer <br>
    - A dropout layer <br>
    - A fully connected layer <br>
    - Another fully connected layer which outputs an 1x4 array containing keypoint predictions [x1,x2,y1,y2]

Train model for 20 epochs. Might be overfit given size of training dataset. 


```python
criterion = nn.L1Loss()
# device = 'cuda:0' ## If you have large enough GPU can uncomment this
device = 'cpu'
num_epochs = 20

## Define model with drouput
model = KeypointDetector(use_dropout=True)
model.to(device)
model = model.double()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

## Define model without dropout
model2 = KeypointDetector(use_dropout=False)
model2.to(device)
model2 = model2.double()
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
```

## Train Model w/o Dropout


```python
model.train_model(train_loader, criterion, optimizer, num_epochs, device)
```

    Epoch [1/20], Loss: 96.6075
    Epoch [2/20], Loss: 89.4517
    Epoch [3/20], Loss: 77.4442
    Epoch [4/20], Loss: 55.2641
    Epoch [5/20], Loss: 66.6324
    Epoch [6/20], Loss: 63.7445
    Epoch [7/20], Loss: 58.3330
    Epoch [8/20], Loss: 44.8522
    Epoch [9/20], Loss: 65.6575
    Epoch [10/20], Loss: 62.5134
    Epoch [11/20], Loss: 60.9977
    Epoch [12/20], Loss: 45.9365
    Epoch [13/20], Loss: 64.2656
    Epoch [14/20], Loss: 58.3298
    Epoch [15/20], Loss: 60.1609
    Epoch [16/20], Loss: 52.7741
    Epoch [17/20], Loss: 53.6427
    Epoch [18/20], Loss: 59.5858
    Epoch [19/20], Loss: 45.5805
    Epoch [20/20], Loss: 45.5334
    Finished Training



    
![png](pizza_walkthrough_files/pizza_walkthrough_20_1.png)
    


## Train Model w/ Dropout


```python
model2.train_model(train_loader, criterion, optimizer2, num_epochs, device)
```

    Epoch [1/20], Loss: 82.3577
    Epoch [2/20], Loss: 88.4738
    Epoch [3/20], Loss: 98.1546
    Epoch [4/20], Loss: 71.0275
    Epoch [5/20], Loss: 66.9632
    Epoch [6/20], Loss: 57.9918
    Epoch [7/20], Loss: 85.6712
    Epoch [8/20], Loss: 79.7051
    Epoch [9/20], Loss: 68.0050
    Epoch [10/20], Loss: 70.7276
    Epoch [11/20], Loss: 63.1107
    Epoch [12/20], Loss: 55.3681
    Epoch [13/20], Loss: 54.8978
    Epoch [14/20], Loss: 56.7542
    Epoch [15/20], Loss: 54.8794
    Epoch [16/20], Loss: 57.0465
    Epoch [17/20], Loss: 51.5654
    Epoch [18/20], Loss: 50.2793
    Epoch [19/20], Loss: 55.0557
    Epoch [20/20], Loss: 51.1545
    Finished Training



    
![png](pizza_walkthrough_files/pizza_walkthrough_22_1.png)
    


## Run Evaluation on Validation Set


```python
e1 = model.evaluate_model(model, test_loader, criterion, device)
e2 = model.evaluate_model(model2, test_loader, criterion, device)

print(f'Performance of first model on test dataset (average loss): {e1}')
print('-'*50)
print(f'Performance of second model on test dataset (average loss): {e2}')
```

    Performance of first model on test dataset (average loss): 102.19229096475998
    --------------------------------------------------
    Performance of second model on test dataset (average loss): 75.57393738579673


## Visualize the Predictions on Untransformed Validation Data


```python
test_0 = transformed_test_dataset.__getitem__(0)
m1_preds = model.predict(model, test_0['image'])
m2_preds = model.predict(model2, test_0['image'])

m1_preds_rescaled = dl.transform_pred_to_normal(m1_preds, (224,224), untransformed_test_dataset.__getitem__(0)['image'])
m2_preds_rescaled = dl.transform_pred_to_normal(m2_preds, (224,224), untransformed_test_dataset.__getitem__(0)['image'])
```


```python
## Model 1

dl.visualize_matrix_with_coordinates(untransformed_test_dataset.__getitem__(0)['image'], m1_preds_rescaled, flip_y=False)

print(f"Angle of slice: {dl.calculate_clockwise_angle(m1_preds_rescaled)}")
```


    
![png](pizza_walkthrough_files/pizza_walkthrough_27_0.png)
    


    Angle of slice: 67.53800581619502



```python
## Model 2

dl.visualize_matrix_with_coordinates(untransformed_test_dataset.__getitem__(0)['image'], m2_preds_rescaled, flip_y=False)

print(f"Angle of slice: {dl.calculate_clockwise_angle(m2_preds_rescaled)}")
```


    
![png](pizza_walkthrough_files/pizza_walkthrough_28_0.png)
    


    Angle of slice: 61.512314946666244


## Conclusions

- Both models perform generally the same.
- On the test dataset the predicted angle is somewhat correct, despite the keypoints not being correct. 


## Improvements

- Larger training set. 
- Potentially exploring different model architectures, pretrained models e.g. ResNet-50 etc. 
