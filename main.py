from LinearRegression import LinearRegression
from LogisticRegression import LogisticRegression
import numpy as np


def display(Y):
    print(f'y_hat : {Y}')
if __name__=="__main__":
    print("L E A R N I N G  A L G O R I T H M S \nPress  :\n 1 -> Linear Regression\n 2 -> LogisticRegression\n")
    choice = input()
    model=None
    match choice:
        case '1' :
            model=LinearRegression()
            X=np.array([[1,1],[2,2],[3,3],[4,4],[5,5]])
            y=np.array([3,5,7,9,11])
            model.fit(X,y)
            display(model.predict([[120,130],[7,7]]))
            
        case '2': 
            model=LogisticRegression()
            X = np.array([[100, 12], [120, 20], [115, 11], [90, 8], [150, 22]])
            y = np.array([0, 1, 0, 0, 1])
            model.fit(X,y)
            display(model.predict([[100,13],[165,20]]))
            