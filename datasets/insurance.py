import torch
import numpy as np
import pandas as pd
from .base_dataset import BaseDataset
import sys
sys.path.append("..")
from utils import to_numeric
import pickle
from sklearn.model_selection import train_test_split


class INSURANCE(BaseDataset):

    def __init__(self, name='INSURANCE', single_bit_binary=False, device='cpu', random_state=61, train_test_ratio=0.2):
        super(INSURANCE, self).__init__(name=name, device=device, random_state=random_state)

        self.train_test_ratio=train_test_ratio
        self.features = {
            'Month': ['Dec', 'Jan', 'Oct', 'Jun', 'Feb', 'Nov', 'Apr', 'Mar', 'Aug', 'Jul', 'May', 'Sep'],
            'WeekOfMonth': None,
            'DayOfWeek': ['Wednesday', 'Friday', 'Saturday', 'Monday', 'Tuesday', 'Sunday', 'Thursday'],
            'Make': ['Honda', 'Toyota', 'Ford', 'Mazda', 'Chevrolet', 'Pontiac', 'Accura', 'Dodge', 'Mercury', 
                     'Jaguar', 'Nisson', 'VW', 'Saab', 'Saturn', 'Porche', 'BMW', 'Mecedes', 'Ferrari', 'Lexus'],
            'AccidentArea': ['Urban', 'Rural'],
            'DayOfWeekClaimed': ['Tuesday', 'Monday', 'Thursday', 'Friday', 'Wednesday', 'Saturday', 'Sunday', '0'],
            'MonthClaimed': ['Jan', 'Nov', 'Jul', 'Feb', 'Mar', 'Dec', 'Apr', 'Aug', 'May', 'Jun', 'Sep', 'Oct', '0'],
            'WeekOfMonthClaimed': None,
            'Sex': ['Female', 'Male'],
            'MaritalStatus': ['Single', 'Married', 'Widow', 'Divorced'],
            'Age': None,
            'Fault': ['Policy Holder', 'Third Party'],
            'PolicyType': ['Sport - Liability', 'Sport - Collision', 'Sedan - Liability', 'Utility - All Perils',
                            'Sedan - All Perils', 'Sedan - Collision', 'Utility - Collision', 'Utility - Liability', 
                            'Sport - All Perils'],
            'VehicleCategory': ['Sport', 'Utility', 'Sedan'],
            'VehiclePrice': ['more than 69000', '20000 to 29000', '30000 to 39000', 'less than 20000', '40000 to 59000', '60000 to 69000'],
            'RepNumber': None,
            'Deductible': None,
            'DriverRating': None,
            'Days_Policy_Accident': ['more than 30', '15 to 30', 'none', '1 to 7', '8 to 15'],
            'Days_Policy_Claim': ['more than 30', '15 to 30', '8 to 15', 'none'],
            'PastNumberOfClaims': ['none', '1', '2 to 4', 'more than 4'],
            'AgeOfVehicle': ['3 years', '6 years', '7 years', 'more than 7', '5 years', 'new', '4 years', '2 years'],
            'AgeOfPolicyHolder': ['26 to 30', '31 to 35', '41 to 50', '51 to 65', '21 to 25', '36 to 40', '16 to 17', 'over 65', '18 to 20'],
            'PoliceReportFiled': ['No', 'Yes'],
            'WitnessPresent': ['No', 'Yes'],
            'AgentType': ['External', 'Internal'],
            'NumberOfSuppliments': ['none', 'more than 5', '3 to 5', '1 to 2'],
            'AddressChange_Claim': ['1 year', 'no change', '4 to 8 years', '2 to 3 years', 'under 6 months'],
            'NumberOfCars': ['3 to 4', '1 vehicle', '2 vehicles', '5 to 8', 'more than 8'],
            'Year': None,
            'BasePolicy': ['Liability', 'Collision', 'All Perils'],
            'FraudFound_P': [0, 1],
        }

        self.single_bit_binary = single_bit_binary
        self.label = 'FraudFound_P'

        self.train_features = {key: self.features[key] for key in self.features.keys() if key != self.label}

        data_df = pd.read_csv('datasets/INSURANCE/Insurance_claims.csv', delimiter=',', names=list(self.features.keys()), skiprows=1, engine='python')

        self.raw = data_df

        data = data_df.to_numpy()

        data_num = to_numeric(data, self.features, label=self.label, single_bit_binary=self.single_bit_binary)

        # split labels and features
        X, y = data_num[:, :-1], data_num[:, -1]
        self.num_features = X.shape[1]

        # create a train and test split and shuffle
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=self.train_test_ratio,
                                                        random_state=self.random_state, shuffle=True)

        # convert to torch
        self.Xtrain, self.Xtest = torch.tensor(Xtrain).to(self.device), torch.tensor(Xtest).to(self.device)
        self.ytrain, self.ytest = torch.tensor(ytrain, dtype=torch.long).to(self.device), torch.tensor(ytest, dtype=torch.long).to(self.device)

        # set to train mode as base
        self.train()

        # calculate the standardization statistics
        self._calculate_mean_std()

        # calculate the histograms and feature bounds
        self._calculate_categorical_feature_distributions_and_continuous_bounds()