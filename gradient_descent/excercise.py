import numpy as np 
import pandas as pd
import math
from sklearn.linear_model import LinearRegression

def gradient(x,y):
    m_curr = b_curr =0
    n = len(x)
    learning_rate = 0.0002
    cs_score_prev = 0
    i=1
    
    while i:
        yp = m_curr*x + b_curr
        cs_score = (1/n) * sum([value ** 2 for value in (y-yp)])
        if math.isclose(cs_score, cs_score_prev, rel_tol=1e-20):
            print(f"m {m_curr} , b {b_curr} , cs_score {cs_score_prev}")
            print("GOT The Value " , i)
            break
        dm = (-2/n) * sum(x*(y - yp))
        bd = (-2/n) * sum((y-yp))
        m_curr = m_curr - learning_rate * dm
        b_curr = b_curr - learning_rate * bd
        cs_score_prev = cs_score
        print(f"m {m_curr} , b {b_curr} , cs_score {cs_score}")
        i=i+1
    pass



df = pd.read_csv("test_scores.csv")
x=np.array(df.math)
y=np.array(df.cs)
gradient(x,y)


def predict_using_sklearn():
    df = pd.read_csv("test_scores.csv")
    r = LinearRegression()
    r.fit(df[['math']],df.cs)
    print(f"r.coef_  {r.coef_} ,  r.intercept_ {r.intercept_}")
    
predict_using_sklearn()