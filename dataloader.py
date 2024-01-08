import sqlite3
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pandas as pd
import math
import torch
import random
from torch.utils.data import Dataset

class create_data(object):
    def __init__(self, data_path, table_name, ):
        self.conn = sqlite3.connect(data_path)
        self.table_name = table_name
        self.data_path = data_path
        self.pic_path = data_path.split('/')[1]

    def split_data(self, train_fraction, df_schema, seed=23):
        '''
        Splits data using Python's random seed for reproducibility.

        Returns train and valid sets as pandas DFs

        Inputs: 
        train_fraction = decimal value 
        df_schema = the column headers
        seed = random seed for reproducibility (default: 42)

        Outputs:
        training_set, validation_set = pandas dataframes
        '''
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name};")
        total_rows = cursor.fetchone()[0]

        # Set the seed and generate indices
        random.seed(seed)
        indices = list(range(total_rows))
        random.shuffle(indices)

        train_size = int(total_rows * train_fraction)
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:]

        train_indices_str = ','.join([str(i + 1) for i in train_indices])  # Assuming row indexing starts at 1
        cursor.execute(f"SELECT * FROM {self.table_name} WHERE rowid IN ({train_indices_str});")
        training_set = cursor.fetchall()

        valid_indices_str = ','.join([str(i + 1) for i in valid_indices])  # Assuming row indexing starts at 1
        cursor.execute(f"SELECT * FROM {self.table_name} WHERE rowid IN ({valid_indices_str});")
        validation_set = cursor.fetchall()

        cursor.close()

        training_set = pd.DataFrame(training_set, columns=df_schema)
        validation_set = pd.DataFrame(validation_set, columns=df_schema)

        return training_set, validation_set

    # def split_data(self, train_fraction, df_schema):
    #     '''
    #     Splits data based on a provided input split fraction

    #     Returns train and valid sets as pandas DFs

    #     Inputs: 
    #     train_fraction = decimal value 
    #     df_schema = the column headers

    #     Outputs:
    #     training_set, validation_set = pandas dataframes
    #     '''
    #     cursor = self.conn.cursor()
    #     cursor.execute(f"SELECT COUNT(*) FROM {self.table_name};")
    #     total_rows = cursor.fetchone()[0]
    #     train_size = int(total_rows * train_fraction)
    #     cursor.execute(f"SELECT * FROM {self.table_name} ORDER BY RANDOM() LIMIT {train_size};")
    #     training_set = cursor.fetchall()
    #     cursor.execute(f"SELECT * FROM {self.table_name} ORDER BY RANDOM() LIMIT {total_rows - train_size} OFFSET {train_size};")
    #     validation_set = cursor.fetchall()
    #     cursor.close()
    #     training_set = pd.DataFrame(training_set, columns=df_schema)
    #     validation_set = pd.DataFrame(validation_set, columns=df_schema)
    #     return training_set, validation_set

    def visualize_matrix(self, matrix):
        '''
        Visualizes the matrix without coordinates

        Inputs:
        matrix = the numpy matrix returned from jpeg_to_matrix function

        Outputs:
        None
        '''
        plt.imshow(matrix)
        plt.show()

    # def visualize_matrix_with_coordinates(self, matrix, coordinates):
    #     '''
    #     Visualizes the matrix with coordinates

    #     Inputs:
    #     matrix = the numpy matrix or tensor
    #     coordinates = a list of tuples corresponding to the XY values of the tip and behind tip

    #     Outputs:
    #     None
    #     '''
    #     if isinstance(matrix, np.ndarray):
    #         plt.imshow(matrix)
    #     elif torch.is_tensor(matrix):
    #         if len(matrix.shape) == 3 and matrix.shape[0] == 1:
    #             matrix = matrix.squeeze(0)  # Remove the singleton dimension
    #         plt.imshow(matrix.numpy())
    #     else:
    #         raise ValueError("Unsupported type for 'matrix'. Use numpy array or PyTorch tensor.")

    #     # x_coords, y_coords = zip(*coordinates)
    #     plt.scatter(coordinates[:, 0], coordinates[:, 1], c='red', marker='x', s=50)
    #     plt.show()

    def visualize_matrix_with_coordinates(self, matrix, coordinates, flip_y=True):
        '''
        Visualizes the matrix with coordinates.

        Inputs:
        matrix = the numpy matrix or tensor
        coordinates = a numpy array of shape (4,) corresponding to [x1, x2, y1, y2]

        Outputs:
        None
        '''
        if isinstance(matrix, np.ndarray):
            plt.imshow(matrix)
        elif torch.is_tensor(matrix):
            if len(matrix.shape) == 3 and matrix.shape[0] == 1:
                matrix = matrix.squeeze(0)  # Remove the singleton dimension
            plt.imshow(matrix.numpy())
        else:
            raise ValueError("Unsupported type for 'matrix'. Use numpy array or PyTorch tensor.")

        # Extract coordinates
        x_coords = [coordinates[0], coordinates[1]]
        y_coords = [coordinates[2], coordinates[3]]

        plt.scatter(x_coords, y_coords, c='red', marker='x', s=50)
        
        if flip_y:
            plt.gca().invert_yaxis()

        plt.show()

    def transform_pred_to_normal(self, predictions, transform_size, original_image):
        
        h, w = transform_size
        orig_h, orig_w = original_image.shape[:2]
        
        scale_x, scale_y = orig_w / w, orig_h / h

        reversed_key_pts = np.array([predictions[0] * scale_x, predictions[1] * scale_x, 
                                     predictions[2] * scale_y, predictions[3] * scale_y])

        return reversed_key_pts


    def calculate_clockwise_angle(self, points):
        """
        Calculate the clockwise angle from the positive x-axis to the line defined by the points in a numpy array.

        Args:
        points: A numpy array of length 4 in the format [x1, x2, y1, y2].

        Returns:
        Angle in degrees, measured clockwise from the positive x-axis, in the range [0, 360).
        """
        m = (points[3] - points[2]) / (points[1] - points[0])
        theta = math.atan(m)
        angle_rad = (math.pi / 2) - theta
        angle_deg = math.degrees(angle_rad)
    
        return angle_deg

class PizzaDataset(Dataset):

    def __init__(self, dataframe, pic_path, transform=None):
        '''
        Inputs:
        csv_file (string): Path to the csv file with annotations.

        root_dir (string): Directory with all the images.

        transform (callable, optional): Optional transform to be applied on a sample.
        '''
        self.key_points_frame = dataframe
        self.transform = transform
        self.pic_path = pic_path

    def __len__(self):
        return len(self.key_points_frame)

    def __getitem__(self, idx):
        '''
        Takes either the train or test dataframes, and a desired index position, then returns the image and a transformed matrix

        If transform True =  will apply various image transformations

        Used for feeding the model

        Inputs 
        key_frame = the dataframe containing the image path and coords
        idx = The desired row of the dataframe 
        transform = Bool. Applies a series of image transforms to the image prior to training.
        '''
        image_name = self.key_points_frame.iloc[idx]['data_path']
        
        image = mpimg.imread(os.path.join(self.pic_path, image_name))
        
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        # key_pts = np.array([[self.key_points_frame.iloc[idx]['x1'],self.key_points_frame.iloc[idx]['y1']],[self.key_points_frame.iloc[idx]['x2'],self.key_points_frame.iloc[idx]['y2']]])
        # key_pts = key_pts.astype('float').reshape(-1, 2)
        key_pts = np.array([self.key_points_frame.iloc[idx]['x1'],self.key_points_frame.iloc[idx]['x2'],self.key_points_frame.iloc[idx]['y1'],self.key_points_frame.iloc[idx]['y2']]).astype('float')
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)
    
        return sample




