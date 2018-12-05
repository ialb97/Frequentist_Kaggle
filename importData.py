#Keene Kaggle Contest data import
import numpy as np
import pandas as pd
import pdb
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from xgboost import XGBRegressor
from keras import regularizers

def getData(trainfile,testfile):
	train_data_features=pd.read_csv(trainfile)

	train_data_features.columns=['ids','RatingID','erkey','AccountabilityID','RatingYear','BaseYear','RatingInterval','CN_Priority',
				  'Creation_Date','date_calculated','publish_date','exclude','ExcludeFromLists','previousratingid',
				  'DataEntryAnalyst','Rpt_Comp_Date','Rpt_Comp_Emp','Reader2_Date','EmployeeID','Rpt_Ap_Date',
				  'Rpt_Ap_Emp','Rpt_Ver_Date','Incomplete','BoardList_Status','StaffList_Status','AuditedFinancial_status',
				  'Form990_status','Privacy_Status','StatusID','DonorAdvisoryDate','DonorAdvisoryText','IRSControlID','ResultsID',
				  'DonorAdvisoryCategoryID','RatingTableID','CauseID','CNVersion','Tax_Year','CEO_Salary','Direct_Support',
				  'Indirect_Support','Govt_Grants','Total_Contributions','ProgSvcRev','MemDues','PrimaryRev','Other_Revenue',
				  'Total_Revenue','Excess_Deficit','Int_Expense','Depreciation','Total_Expenses','Total_Func_Exp','Program_Expenses',
				  'Administration_Expenses','Fundraising_Expenses','Pymt_Affiliates','Total_Assets','Total_Liabilities',
				  'Total_Net_Assets','Assets_45','Assets_46','Assets_47c','Assets_48c','Assets_49','Assets_54','Liability_60']

	train_data_features = train_data_features.fillna(0)

	train_data_features = train_data_features.set_index('ids')

	train_data_features = train_data_features.drop([5449])

	train_data_features = train_data_features.drop(['RatingYear','BaseYear','Creation_Date','date_calculated','publish_date','ExcludeFromLists',
							  'Rpt_Comp_Date','Rpt_Comp_Emp','Reader2_Date','Rpt_Ap_Date','Rpt_Ver_Date','Incomplete',
							  'StatusID','DonorAdvisoryDate','DonorAdvisoryText','ResultsID','DonorAdvisoryCategoryID',
							  'Tax_Year','Direct_Support','Indirect_Support','Int_Expense','Depreciation',
							  'Assets_45','Assets_46','Assets_47c','Assets_48c','Assets_49','Assets_54','Liability_60'],axis=1)

	train_data_features = train_data_features.dropna(axis='columns',how='all')

	train_data_labels = pd.read_csv('trainLabels.csv')

	train_data_labels.columns = ['ids','ATScore','OverallScore']

	train_data_labels = train_data_labels.set_index('ids')

	train_data_labels = train_data_labels.drop([5449],axis=0)

	train_data_labels = train_data_labels.drop(['ATScore'],axis=1)

	train_data_features.erkey = train_data_features.erkey.str.extract('(\d+)', expand=False)

	train_features = train_data_features.values

	scalar = preprocessing.MinMaxScaler()

	train_features = scalar.fit_transform(train_features)

	#train_features = train_features.reshape(train_features.shape[0],train_features.shape[1],1)

	train_labels = train_data_labels.values

	train_labels = train_labels/100

	#train_labels = train_labels.reshape(train_labels.shape[0],train_labels.shape[1],1)

	train_features, val_features, train_labels, val_labels =train_test_split(
		train_features,train_labels,test_size=.1,random_state=2345432)

	test_data_features = pd.read_csv(testfile)

	test_data_features.columns=['ids','RatingID','erkey','AccountabilityID','RatingYear','BaseYear','RatingInterval','CN_Priority',
				  'Creation_Date','date_calculated','publish_date','exclude','ExcludeFromLists','previousratingid',
				  'DataEntryAnalyst','Rpt_Comp_Date','Rpt_Comp_Emp','Reader2_Date','EmployeeID','Rpt_Ap_Date',
				  'Rpt_Ap_Emp','Rpt_Ver_Date','Incomplete','BoardList_Status','StaffList_Status','AuditedFinancial_status',
				  'Form990_status','Privacy_Status','StatusID','DonorAdvisoryDate','DonorAdvisoryText','IRSControlID','ResultsID',
				  'DonorAdvisoryCategoryID','RatingTableID','CauseID','CNVersion','Tax_Year','CEO_Salary','Direct_Support',
				  'Indirect_Support','Govt_Grants','Total_Contributions','ProgSvcRev','MemDues','PrimaryRev','Other_Revenue',
				  'Total_Revenue','Excess_Deficit','Int_Expense','Depreciation','Total_Expenses','Total_Func_Exp','Program_Expenses',
				  'Administration_Expenses','Fundraising_Expenses','Pymt_Affiliates','Total_Assets','Total_Liabilities',
				  'Total_Net_Assets','Assets_45','Assets_46','Assets_47c','Assets_48c','Assets_49','Assets_54','Liability_60']


	test_data_features = test_data_features.fillna(0)

	test_data_features = test_data_features.set_index('ids')

	test_data_features = test_data_features.drop(['RatingYear','BaseYear','Creation_Date','date_calculated','publish_date','ExcludeFromLists',
							  'Rpt_Comp_Date','Rpt_Comp_Emp','Reader2_Date','Rpt_Ap_Date','Rpt_Ver_Date','Incomplete',
							  'StatusID','DonorAdvisoryDate','DonorAdvisoryText','ResultsID','DonorAdvisoryCategoryID',
							  'Tax_Year','Direct_Support','Indirect_Support','Int_Expense','Depreciation',
							  'Assets_45','Assets_46','Assets_47c','Assets_48c','Assets_49','Assets_54','Liability_60'],axis=1)

	test_data_features = test_data_features.dropna(axis='columns',how='all')

	test_data_features.erkey = test_data_features.erkey.str.extract('(\d+)', expand=False)

	test_features = test_data_features.values

	test_features = scalar.fit_transform(test_features)

	return train_features,train_labels,val_features,val_labels,test_features

def DenseLayerModel(train_features,train_labels,val_features,val_labels,test_features):
	BATCH_SIZE = 32
	EPOCHS = 25

	input_shape = (train_features.shape[1],)

	model = Sequential()

	model.add(Dense(32,activation='relu',input_shape=input_shape,kernel_initializer='random_uniform',bias_initializer='zeros'))
	model.add(Dense(64,activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.001)))
	model.add(Dense(128,activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
	model.add(Dense(1,activation='sigmoid',kernel_initializer='random_uniform',bias_initializer='zeros'))

	model.compile(loss=keras.losses.mean_squared_error,
	            	optimizer=keras.optimizers.Adam(),
					metrics=['accuracy'])

	pdb.set_trace()

	model.fit(train_features,train_labels,
				 batch_size=BATCH_SIZE,
				 epochs=EPOCHS,
	          	 verbose=1,
	           	 validation_data=(val_features, val_labels))



	output = model.predict(test_features,batch_size=BATCH_SIZE)

	output = output*100

	print(output)

	output = pd.DataFrame(output)

	output.to_csv('output.csv')

	return True

def xgboostmodel(train_features,train_labels,val_features,val_labels,test_features):
	model = XGBRegressor(max_depth=10,learning_rate=.1,n_estimators=1000,subsample=0.8)
	print("test")
	pdb.set_trace()
	model.fit(train_features, train_labels)
	predictions = model.predict(test_features)
	predictions = predictions*100

	predict = pd.DataFrame(predictions)

	predict.to_csv('predict.csv')
	print(predictions)


if __name__ =="__main__":

	x_train,y_train,x_val,y_val,x_test = getData("trainFeatures.csv","testFeatures.csv")

	x = DenseLayerModel(x_train,y_train,x_val,y_val,x_test)

	#y = xgboostmodel(x_train,y_train,x_val,y_val,x_test)
