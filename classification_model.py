
#pip install streamlit
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 

from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing  import  RobustScaler
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error , mean_squared_error 



# start with classification app
st.title("Classification App")

# slidebar section
#1. upload & red data
st.sidebar.header("Input Data")
file = st.sidebar.file_uploader("Please Upload The Data" , type=["csv", "xls", "xlsx"])
if file is not None:
    file_type = file.name.split('.')[-1].lower()
    if file_type=="csv":
        data = pd.read_csv(file)
        st.success("CSV file uploaded successfully!")
    elif file_type in ["xls", "xlsx"]:
        data = pd.read_excel(file)
        st.success("Excel file uploaded successfully!")
    else:
        st.error("Unsupported file type!")
        data = None
    st.write("The Data" , data.head())
    features  = st.sidebar.multiselect("select features" , data.columns)
    target  = st.sidebar.selectbox("select target" , data.columns) 
    # show data without unused columns
    target = [target]
    data = data[features+target]
    st.write("Selected Data ", data.head())

    # feature engineering step
    st.header("Feature Engineering")
    flag = st.checkbox("Date Feature ")
    if flag:
        user_feature = st.selectbox("Select Feature ", options=data.columns)
        data['hour'] = pd.to_datetime(data[user_feature]).dt.hour
        st.write("Hour Column is" , data['hour'].head())
        data.drop(columns=user_feature,inplace=True)
        st.write("Your Data" , data.head())
    



    #Preprocessing
    st.header("Preprocessing")
    #1. missing values
    col1,col2 = st.columns(2)  
    with col1:  
        missing = data.isna().sum()
        miss = st.checkbox("Check Missing Values")
        if miss:
            st.write("Missing Values" , missing)
        if st.button("Remove Missing Data"):
            data.dropna(inplace=True)
            st.write("✅ Null Data Is Removes :)")
    # 2. duplicated values
    with col2:
        duplicated = data.duplicated().sum()
        dup = st.checkbox("Check Duplicated Values")
        if dup:
            st.write("Duplicated Values" , duplicated)
        if st.button("Remove Dupicated Data"):
            data.drop_duplicates(inplace=True)
            st.write("✅ Dupicated Data Is Removes :)")
    
    # 3. outliers
    outliers = st.checkbox("Check Outliers")     
    if outliers:
        plt.style.use("dark_background")
        fig , ax =  plt.subplots(figsize=(5, 3))
        sns.boxplot(data , ax= ax, color="tomato" ) 
        plt.xticks(rotation=45,ha="right" , size=7)
        plt.yticks( size=7)
        st.pyplot(fig)
        # Outliers Handling
        if_outliers = st.checkbox("Outliers Handling")
        if if_outliers:
            st.subheader("Outliers Handling Options")
            num_cols = data.select_dtypes(include=['float64', 'int64']).columns.to_list()
            col = st.selectbox("Select column for outlier handling",num_cols)
            method = st.radio("Choose handling method:", ["Remove Outliers", "Cap Outliers"])
            st.write(f"Data before handling ({col})")
            st.dataframe(data[[col]].describe())

            # apply processing by using IQR
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR
            
            if method == "Remove Outliers":
                data = data[(data[col]>=lower_limit) & (data[col]<=upper_limit)]
                st.success("✅ Outliers removed successfully!")
                flagy=True
            else: #Cap Outliers
                data[col] = data[col].clip(lower=lower_limit , upper=upper_limit) 
                st.success("✅ Outliers capped successfully!")
                flagy=True
            
            # show final result
            st.write(f"### Data after handling ({col})")
            st.dataframe(data[col].describe())
        
        

    # 4. encoding
    enc = st.checkbox("Categorical Encoding")
    if enc :
        category = data.select_dtypes("object")
        st.write(category)
        st.subheader("Categorical Columns Encoding")
        if len(category.columns) == 0:
            st.warning("⚠ No categorical columns found!")
        else:
            col = st.selectbox("Select categorical column",category.columns)
            st.write(f"Unique values in **{col}**:" , data[col].unique())
            # choose column type
            col_type = st.radio(
                "Select the type of categorical data:",
                ["Nominal (No order)", "Ordinal (Has order)"]
            )
            # if nomina -->  One-Hot Encoding
            if col_type=="Nominal (No order)":
                st.info("Recommended method: **One-Hot Encoding**")
                data = pd.get_dummies(data , columns=[col] , drop_first=False)
                t_flag = True
                st.success(f"✅ Applied One-Hot Encoding on {col}")
                flagy=True
                st.dataframe(data.head())
            else:
                st.info("Recommended method: **Label Encoding**")
                label_encoder = LabelEncoder()
                data[col] = label_encoder.fit_transform(data[col])
                t_flag = True
                st.success(f"✅ Applied Label Encoding on {col}")
                flagy=True
                st.dataframe(data.head())
            
            # show final result
            st.write("Final Data After Encoding")
            if col_type =="Nominal (No order)":
                st.dataframe(data.head())
            else:
                st.dataframe(data.head())
   

    # imbalanced
    imb = st.checkbox("Check Balanced data")
    
            
    if imb:
        if len(target) != 1:
            st.error("⚠️ Please select exactly one target column!")
            xtrain, xtest, ytrain, ytest = None, None, None, None
        else:
            target = target[0]
            st.write("Class Distribution Before Oversampling 📊")
            st.write(data[target].value_counts())
            oversample_btn = st.button("Apply Oversampling")
            if oversample_btn:
                # split
                if flagy:
                    x = data.drop(columns=[target])
                    y = data[target]
                    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
                    smote = SMOTE(random_state=42)
                    xtrain = xtrain.apply(pd.to_numeric, errors='coerce')
                    nan_indices = xtrain[xtrain.isna().any(axis=1)].index
                    xtrain = xtrain.drop(nan_indices)
                    ytrain = ytrain.drop(nan_indices)
                    X_resampled, y_resampled = smote.fit_resample(xtrain , ytrain)

                    # show results
                    st.write("✅smote is applied sicessfully :)")
                    st.write("data distribution after process 📊")
                    st.write(y_resampled.value_counts())

                    # save new data
                    resamble_data = X_resampled.copy()
                    resamble_data[target]=y_resampled
                    st.write(resamble_data.head())
    # scaling
    x = data.drop(columns=[target])
    y = data[target]
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    scaler = RobustScaler()
    sc=st.checkbox("Scaling The Data")
    if sc:
        num= st.multiselect("Select Numeric Features" , xtrain.columns)
        if num:
            xtrain[num] = scaler.fit_transform(xtrain[num])
            xtest[num] = scaler.transform(xtest[num])
            st.write("✅ Transformation DONE :)")
    # modeling
    # 1. KNN , SVM , LR
    
    model=st.checkbox("Modeling")
    if model:
        selected_model = st.selectbox("Select the model" , options=['KNN' , 'SVM' , 'LR'])
        if selected_model=='KNN':
            k = st.slider("n_neighbors" , 2, 20 , 10)
            weights = st.selectbox("weights" , options= ["uniform" , "distance"])
            metric =   st.selectbox("metric" , options= ["manhattan_distance" , "euclidean_distance"])  
            if metric == "manhattan_distance" : p= 1
            elif metric == "euclidean_distance" : p= 2
            model=  KNeighborsClassifier(n_neighbors= k , weights= weights , p=p )
            
        elif selected_model=='SVM':
            c  = st.number_input("C" , 0.0001 , max_value= 100.0)
            kernal = st.selectbox("weights" , options= ["linear" , "poly" , "rbf"])

            if kernal == "rbf": 
                gamma  = st.number_input("gamma" , 0.0001 , max_value= 100.0)
                model = SVC(kernel= kernal , C = c , gamma=gamma)    
            elif kernal == "poly" : 
                d =     st.slider("degree" , 2, 10 , 5 , step= 1)
                model = SVC(kernel= kernal , C = c , degree= d)   

            elif kernal == "linear" : 
                    model = SVC(kernel= kernal , C = c ) 
            
        elif selected_model=='LR':
                
                penalty = st.selectbox("penalty" , options= ["l1" , "l2" ])
                C = st.number_input("c" , 0.0001 , max_value= 100.0)
                solver = st.selectbox("solver" , options= ["lbfgs" , "sag" ])
                model  = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
        
        
        if t_flag :
            imputer = SimpleImputer(strategy="most_frequent")
            xtrain = pd.DataFrame(imputer.fit_transform(xtrain) , columns=xtrain.columns)
            xtest = pd.DataFrame(imputer.transform(xtest), columns=xtest.columns)
            model.fit(xtrain  , ytrain) 
            ypred = model.predict(xtest) 

            # evaluation 
            st.header("evaluation")
            st.write("classification report" , accuracy_score(ytest , ypred))   



   


                        

                    












        
