import pandas as pd
from sklearn import tree
from IPython.display import Image,display
from sklearn.externals.six import StringIO
import pydotplus
from sklearn.ensemble import RandomForestClassifier

def main():
    data=pd.read_csv("Employee-Attrition.csv",usecols=[1,2,4,6,10,11,13,14,16,17,20,21,22,24,25,28,30,31,33])
    data=preprocess(data)
    constructGraph(data)
    predict(data)

def preprocess(data):
    d={'Yes':1,'No':0}
    data['Attrition']=data['Attrition'].map(d)
    data['OverTime']=data['OverTime'].map(d)
    d={'Non-Travel':0,'Travel_Rarely':1,'Travel_Frequently':2}
    data['BusinessTravel']=data['BusinessTravel'].map(d)
    d={'Sales':0,'Research & Development':1,'Human Resources':2}
    data['Department']=data['Department'].map(d)
    d={'Female':0,'Male':1}
    data['Gender']=data['Gender'].map(d)
    d={'Single':0,'Married':1,'Divorced':2}
    data['MaritalStatus']=data['MaritalStatus'].map(d)
    d = {'Y': 1, 'N': 0}
    data['Over18']=data['Over18'].map(d)
    return data

def constructGraph(data):
    features=list(data.columns[1:19])
    print (features)
    y=data['Attrition']
    x=data[features]
    dec=tree.DecisionTreeClassifier()
    dec=dec.fit(x,y)
    dot_data = StringIO()
    tree.export_graphviz(dec, out_file=dot_data,
                         feature_names=features)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    display(Image(graph.create_png()))


def predict(data):
    features = list(data.columns[1:19])
    y = data['Attrition']
    x = data[features]
    dec=RandomForestClassifier(n_estimators=10)
    dec=dec.fit(x,y)
    print(dec.predict([[1,0,2,2,0,3,2,4,0,8,1,1,3,1,8,1,6,0]]))
    print(dec.predict([[2,1,1,3,1,2,2,2,1,1,1,0,4,4,10,3,10,1]]))
    print(dec.predict([[1,1,2,4,1,2,1,3,0,6,1,1,3,2,7,3,0,0]]))
    print(dec.predict([[2,1,4,4,0,3,1,3,1,1,1,1,3,3,8,3,8,3]]))

if __name__=="__main__" :
    main()