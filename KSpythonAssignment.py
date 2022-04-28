"""Python Assignment for Programming with Python
Student Name: Korawin Suwunnapuk

There are 3 datasets provided which are
A) 4 Training functions
B) 1 test dataset
C) 50 ideal functions
All data consists of x-y-pairs of values"""

# Firstly, declare all libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sqlalchemy as db
import mysql.connector as mysql
from mysql.connector import Error

# Write a class to use inheritance to plot graph results
class plotgraph(object):
    def __init__(self, x_value1=None, y_value1=None, color1=None, x_value2=None, y_value2=None, color2=None, legend1=None, legend2=None):
        self.x_value1 = x_value1
        self.y_value1 = y_value1
        self.x_value2 = x_value2
        self.y_value2 = y_value2
        self.color1 = color1
        self.color2 = color2
        self.legend1 = legend1
        self.legend2 = legend2
        
    def plotresult(self):
        plt.scatter(self.x_value1, self.y_value1, color = self.color1 )
        plt.scatter(self.x_value2, self.y_value2, color = self.color2 )
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend([self.legend1, self.legend2], loc="upper right")
        plt.show()

class plot1(plotgraph):
    tdlegend1 = "Training data"
    tdlegend2 = "Ideal function"
    def __init__(self, x_value1, y_value1, color1, x_value2, y_value2, color2):
        super().__init__(x_value1, y_value1, color1, x_value2, y_value2, color2, self.tdlegend1, self.tdlegend2)
    
   
# All calculations in this Class will be tested by Unit test module (in test_KSpythonAssignment file)
class calculations(object):
    def leastsquare(self, observed, fitted):
        return (observed-fitted)**2
    def MSEdev(self, observed, estimated):
        mse = (observed-estimated)**2/100
        return mse
    def RMSEdev(self, observed, estimated): 
        rmse1 = (observed-estimated)**2/100
        rmse = np.sqrt(rmse1)
        return rmse
    def MAEdev(self, observed, estimated):
        mae = (observed-estimated)/100
        return abs(mae)
    def subtraction(self, a, b):
        return a-b

# User-Defined Exceptions
class MyException(Exception):
    def __init___(self, exception_message):
        super().__init__(self, exception_message)


def main():

    #######################################################################################
    # I have created a database name "ksassignment" to use with this Python assignment
    """     The database information are
            database name:ksassignment
            host:localhost
            user:root
            password:12345                  """ 
    #######################################################################################

    # Establish the connection to the database
    engine = db.create_engine("mysql://root:12345@localhost/ksassignment")
    connection = engine.connect()

    meta_train = db.MetaData()
    meta_ideal = db.MetaData()

    # Establish the connection to MYSQL
    conn = mysql.connect(host='localhost', database='ksassignment', user='root', password='12345')
    cursor = conn.cursor()
    cursor.execute("select database();")
    record = cursor.fetchone()
    if conn.is_connected():
        print("\nYou're connected to database: ", record)
    else:
        raise MyException("Cannot connect to database {}".format(record))
    
    # Read the given csv data files
    train = pd.read_csv("train.csv")
    ideal = pd.read_csv("ideal.csv")
    test = pd.read_csv("test.csv")
    # Create DataFrame from each file 
    df_train = pd.DataFrame(train)
    df_ideal = pd.DataFrame(ideal)

    x_train = train.x.tolist()
    x_ideal = ideal.x.tolist()

    # Create Table1 for The training data in the database
    Table1 = db.Table("Table1_The_training_data", meta_train,
    db.Column("X", db.Float, nullable=False),
    db.Column("Y1(training func)", db.Float, nullable=False),
    db.Column("Y2(training func)", db.Float, nullable=False),
    db.Column("Y3(training func)", db.Float, nullable=False),
    db.Column("Y4(training func)", db.Float, nullable=False))
    meta_train.create_all(engine)

    # Delete existing data in Table 1 in the the database
    sql_query_delete_train = db.delete(Table1)
    result_delete_train = connection.execute(sql_query_delete_train)

    # Insert train Dataframe into Table 1 in the database
    Table1 = db.Table("Table1_The_training_data",meta_train, autoload=True, autoload_with=engine)
    sql_query_train = db.insert(Table1)

    for row in df_train.itertuples():
        fill_table1 = [{"X":row.x,
        "Y1(training func)":row.y1, "Y2(training func)":row.y2,
        "Y3(training func)":row.y3, "Y4(training func)":row.y4}]
        result_load_train = connection.execute(sql_query_train, fill_table1)

    print("\nTable 1 : The training data, is now created in the database.")

    # Create Table2 for The ideal functions in the database
    Table2 = db.Table("Table2_The_ideal_functions", meta_ideal,
    db.Column("X", db.Float, nullable=False),
    db.Column("Y1(ideal func)", db.Float, nullable=False),db.Column("Y2(ideal func)", db.Float, nullable=False),
    db.Column("Y3(ideal func)", db.Float, nullable=False),db.Column("Y4(ideal func)", db.Float, nullable=False),
    db.Column("Y5(ideal func)", db.Float, nullable=False),db.Column("Y6(ideal func)", db.Float, nullable=False),
    db.Column("Y7(ideal func)", db.Float, nullable=False),db.Column("Y8(ideal func)", db.Float, nullable=False),
    db.Column("Y9(ideal func)", db.Float, nullable=False),db.Column("Y10(ideal func)", db.Float, nullable=False),
    db.Column("Y11(ideal func)", db.Float, nullable=False),db.Column("Y12(ideal func)", db.Float, nullable=False),
    db.Column("Y13(ideal func)", db.Float, nullable=False),db.Column("Y14(ideal func)", db.Float, nullable=False),
    db.Column("Y15(ideal func)", db.Float, nullable=False),db.Column("Y16(ideal func)", db.Float, nullable=False),
    db.Column("Y17(ideal func)", db.Float, nullable=False),db.Column("Y18(ideal func)", db.Float, nullable=False),
    db.Column("Y19(ideal func)", db.Float, nullable=False),db.Column("Y20(ideal func)", db.Float, nullable=False),
    db.Column("Y21(ideal func)", db.Float, nullable=False),db.Column("Y22(ideal func)", db.Float, nullable=False),
    db.Column("Y23(ideal func)", db.Float, nullable=False),db.Column("Y24(ideal func)", db.Float, nullable=False),
    db.Column("Y25(ideal func)", db.Float, nullable=False),db.Column("Y26(ideal func)", db.Float, nullable=False),
    db.Column("Y27(ideal func)", db.Float, nullable=False),db.Column("Y28(ideal func)", db.Float, nullable=False),
    db.Column("Y29(ideal func)", db.Float, nullable=False),db.Column("Y30(ideal func)", db.Float, nullable=False),
    db.Column("Y31(ideal func)", db.Float, nullable=False),db.Column("Y32(ideal func)", db.Float, nullable=False),
    db.Column("Y33(ideal func)", db.Float, nullable=False),db.Column("Y34(ideal func)", db.Float, nullable=False),
    db.Column("Y35(ideal func)", db.Float, nullable=False),db.Column("Y36(ideal func)", db.Float, nullable=False),
    db.Column("Y37(ideal func)", db.Float, nullable=False),db.Column("Y38(ideal func)", db.Float, nullable=False),
    db.Column("Y39(ideal func)", db.Float, nullable=False),db.Column("Y40(ideal func)", db.Float, nullable=False),
    db.Column("Y41(ideal func)", db.Float, nullable=False),db.Column("Y42(ideal func)", db.Float, nullable=False),
    db.Column("Y43(ideal func)", db.Float, nullable=False),db.Column("Y44(ideal func)", db.Float, nullable=False),
    db.Column("Y45(ideal func)", db.Float, nullable=False),db.Column("Y46(ideal func)", db.Float, nullable=False),
    db.Column("Y47(ideal func)", db.Float, nullable=False),db.Column("Y48(ideal func)", db.Float, nullable=False),
    db.Column("Y49(ideal func)", db.Float, nullable=False),db.Column("Y50(ideal func)", db.Float, nullable=False))
    meta_ideal.create_all(engine)

    # Delete existing data in Table 2 in the database
    sql_query_delete_ideal = db.delete(Table2)
    result_delete_ideal = connection.execute(sql_query_delete_ideal)

    # Insert ideal Dataframe into Table 2 in the database
    Table2 = db.Table("Table2_The_ideal_functions",meta_ideal, autoload=True, autoload_with=engine)
    sql_query_ideal = db.insert(Table2)

    for row in df_ideal.itertuples():
        fill_table2 = [{"X":row.x,"Y1(ideal func)":row.y1,"Y2(ideal func)":row.y2, "Y3(ideal func)":row.y3,
        "Y4(ideal func)":row.y4, "Y5(ideal func)":row.y5, "Y6(ideal func)":row.y6, "Y7(ideal func)":row.y7, 
        "Y8(ideal func)":row.y8, "Y9(ideal func)":row.y9, "Y10(ideal func)":row.y10, "Y11(ideal func)":row.y11, 
        "Y12(ideal func)":row.y12, "Y13(ideal func)":row.y13, "Y14(ideal func)":row.y14, "Y15(ideal func)":row.y15, 
        "Y16(ideal func)":row.y16, "Y17(ideal func)":row.y17, "Y18(ideal func)":row.y18, "Y19(ideal func)":row.y19, 
        "Y20(ideal func)":row.y20, "Y21(ideal func)":row.y21, "Y22(ideal func)":row.y22, "Y23(ideal func)":row.y23, 
        "Y24(ideal func)":row.y24, "Y25(ideal func)":row.y25, "Y26(ideal func)":row.y26, "Y27(ideal func)":row.y27, 
        "Y28(ideal func)":row.y28, "Y29(ideal func)":row.y29, "Y30(ideal func)":row.y30, "Y31(ideal func)":row.y31, 
        "Y32(ideal func)":row.y32, "Y33(ideal func)":row.y33, "Y34(ideal func)":row.y34, "Y35(ideal func)":row.y35, 
        "Y36(ideal func)":row.y36, "Y37(ideal func)":row.y37, "Y38(ideal func)":row.y38, "Y39(ideal func)":row.y39, 
        "Y40(ideal func)":row.y40, "Y41(ideal func)":row.y41, "Y42(ideal func)":row.y42, "Y43(ideal func)":row.y43, 
        "Y44(ideal func)":row.y44, "Y45(ideal func)":row.y45, "Y46(ideal func)":row.y46, "Y47(ideal func)":row.y47, 
        "Y48(ideal func)":row.y48, "Y49(ideal func)":row.y49, "Y50(ideal func)":row.y50}]
        result_load_ideal = connection.execute(sql_query_ideal, fill_table2)

    print("Table 2 : The ideal functions, is now created in the database.\n")

    """ Step 1 : Use Training data to choose the four Ideal functions

    Use Training function to choose 4 ideal functions which are the best fit out of 50 provided
    by calculating the sum of all y-deviations squared (Least-Square) and choose the minimum one"""

    # Each Training function chooses its best fit ideal function

    s=ideal.shape
    n=s[1] # n-1 is number of Y columns only

    devsquare_train=pd.DataFrame() #create dataframe of y deviation squared
    df_dev=pd.DataFrame() #create dataframe of y deviation squared with summation of the deviation at the end
    best_ideal = [] #creat a list of 4 best fit ideal functions

    for j in range(4):
        for i in range(n-1):
            caldev = calculations()
            dev = caldev.leastsquare(train.iloc[:,j+1], ideal.iloc[:,(i+1)])
            devsquare_train["y"+str(i+1)]=dev

        total = devsquare_train   
        total.loc["total"] = devsquare_train.sum() #find summation of all y deviations in the function   
        
        # find index(column) of minimum value of y-deviations squared
        columindex = total.idxmin(axis = 1)
        val = np.array(columindex)[400]
        print("Best fit Ideal function for the Training function",str(j+1)," is",val)
        best_ideal.append(val)
    
    # Create a dataframe of 4 best fit ideal functions
    four_ideal_functions=ideal[[best_ideal[0],best_ideal[1],best_ideal[2],best_ideal[3]]].copy()

    ###################################################################
    # #                      Ploting graph Set 1                     ##
    #   Plot graph of each training function and its ideal function
    #   You will see 4 graphs here
    
    best=[]
    y_bestfit = []
    y_train =[]
    for i in range(4):  
        best.append(best_ideal[i])
        y_bestfit.append(ideal[best[i]].values)
        y_train.append(train["y"+str(i+1)].values)
    
    for i in range(4):   
        plt.title("Training function {} and its ideal function".format(i+1))
        set1plot=plot1(x_train,y_train[i],"blue",x_train,y_bestfit[i],"red")   
        set1plot.plotresult()  
    ###################################################################
    

    """ Step 2 : Mapping the individual test case to the four ideal functions """

    # Determine for each and every x-y-pair of test data values whether or not they can be assigned to the four chosen ideal functions
    # Then create a dataframe for only mapped point with four ideal functions.
    s1=test.shape
    n=s1[0]
    test_ideal_assigned=pd.DataFrame(columns=best_ideal)

    for i in range(n):
        if test.iloc[i,0] in ideal.iloc[:,0].values:
            findindex = ideal.iloc[:,0].tolist()
            #find which index of test.iloc in findindex 
            index = findindex.index(test.iloc[i,0])
            fill_assigned=[[four_ideal_functions.iloc[index,0],four_ideal_functions.iloc[index,1],four_ideal_functions.iloc[index,2],four_ideal_functions.iloc[index,3]],]
            map=pd.DataFrame(data=fill_assigned,columns=best_ideal)
            test_ideal_assigned=pd.concat([test_ideal_assigned, map],ignore_index=True)

    # Create new train dataframe for only mapped point 
    col_train=["y1","y2","y3","y4"]
    train_assigned=pd.DataFrame(columns=col_train)

    for i in range(n):
        if test.iloc[i,0] in train.iloc[:,0].values: 
            findindex = train.iloc[:,0].tolist() #change whole x column to list
            index = findindex.index(test.iloc[i,0]) # find index that contains x value

            fill_assigned=[[train.iloc[index,1],train.iloc[index,2],train.iloc[index,3],train.iloc[index,4]],]
            map=pd.DataFrame(data=fill_assigned,columns=col_train)
            train_assigned=pd.concat([train_assigned, map],ignore_index=True)

    # Finding maximum deviation for each Ytrain and its ideal function

    deviation = pd.DataFrame(index=np.arange(n), columns = ["y1&its ideal","y2&its ideal","y3&its ideal","y4&its ideal"])

    for j in range(4):
        for i in range(n): #looping through each element of the list
            #fill deviation dataframe with result
            caldev = calculations()
            dev = caldev.MSEdev(train_assigned.iloc[i,j], test_ideal_assigned.iloc[i,j])
            deviation.iat[i,j] = dev 
            # I am using MSE as deviation. If you want to see the result by using other deviations you can change 
            # the function to use RMSE (change function name to RMSEdev), MAE (change function name to MAEdev) instead 
            # as below code as example to compare the result. 
            """caldev = calculations()
            dev = caldev.RMSEdev(train_assigned.iloc[i,j], test_ideal_assigned.iloc[i,j])
            deviation.iat[i,j] = dev"""

           

    max_dev = deviation.max()

    """ Finding largest deviation between test dataset and the ideal function
        Then creating the dataframe for Deviation between (Y test data point) and (Y ideal function)"""

    id1 = best_ideal[0]
    id2 = best_ideal[1]
    id3 = best_ideal[2]
    id4 = best_ideal[3]
    test_ideal_dev = pd.DataFrame(index=np.arange(n), columns =[id1,id2,id3,id4])

    for j in range(n):
        for i in range(4):
            #fill deviation dataframe with result
            caldev = calculations()
            dev = caldev.MSEdev(test.iloc[j,1], test_ideal_assigned.iloc[j,i])
            test_ideal_dev.iat[j,i] = dev
            # I am using MSE as deviation. If you want to see the result by using other deviations you can change 
            # the function to use RMSE (change function name to RMSEdev), MAE (change function name to MAEdev) instead 
            # as below code as example to compare the result. 
            """caldev = calculations()
            dev = caldev.RMSEdev(test.iloc[j,1], test_ideal_assigned.iloc[j,i])
            test_ideal_dev.iat[j,i] = dev"""

    """ Mapping the individual test case to four ideal functions 
        by choosing the ideal function which its largest dev is less than maximum deviation"""

    mapping = pd.DataFrame(index=np.arange(n), columns =["Ideal function 1","Ideal function 2","Ideal function 3","Ideal function 4"])

    for j in range(4):
        for i in range(n):
            if abs(test_ideal_dev.iloc[i,j]) <= abs(max_dev[j]*np.sqrt(2)):
                mapping.iat[i,j]= "YES"
            else:
                mapping.iat[i,j]= "no"

    # Create a table of the different between max deviation and largest deviation
    dev_of_maxdev_largedev= pd.DataFrame(columns =["X test",id1,id2,id3,id4])
    x_test = test.x.tolist()
    dev_of_maxdev_largedev.iloc[:,0]= x_test
        
    for j in range(4):
        for i in range(n):
            caldev = calculations()
            dev = caldev.subtraction(abs(test_ideal_dev.iloc[i,j]), abs(max_dev[j]))
            dev_of_maxdev_largedev.iloc[i,j+1] = dev
            
        
    # Create new dataframe with only deviation value
    new = dev_of_maxdev_largedev[[id1,id2,id3,id4]].copy()
    # Convert series (or columns) to numeric types to be able to find the minimum value in next step
    new[id1] = pd.to_numeric(new[id1])
    new[id2] = pd.to_numeric(new[id2])
    new[id3] = pd.to_numeric(new[id3])
    new[id4] = pd.to_numeric(new[id4])

    # Find index of minimum value in each row
    minValueIndexObj = new.idxmin(axis = 1)

    """ Check if the individual test case is mapping more than one ideal function. 
        If so, choose the one with minimum deviation.
        Then fill mapped point to the result table"""
  
    dev_of_testpoint_ideal= pd.DataFrame(index=np.arange(n),columns =[id1,id2,id3,id4])
    x_test = test.x.tolist()
    for j in range(4):
        for i in range(n):
            caldev = calculations()
            dev = caldev.subtraction(test.iloc[i,1], test_ideal_assigned.iloc[i,j])
            dev_of_testpoint_ideal.iloc[i,j] = dev
        
    # Fill Result table with the deviation and its mapped ideal function
    result = pd.DataFrame(index=np.arange(100), columns =["Xtest","Ytest","DeltaY","noideal"])
    x_test = test.x.tolist()
    y_test = test.y.tolist()
    result.iloc[:,0]= x_test
    result.iloc[:,1]= y_test
    
    for i in range(n):
        if "YES" in mapping.values[i] :
            result.iloc[i,3] = minValueIndexObj[i]
            result.iloc[i,2] = dev_of_testpoint_ideal[minValueIndexObj[i]][i]
    
        else :
            result.iloc[i,3] = "non"
            minimumDev = dev_of_testpoint_ideal.min(axis=1) 
            result.iloc[i,2] = minimumDev[i]

    df_result = pd.DataFrame(result)
    
    # List all mapped test points for Second ideal function
    first = []
    for i in range(n):
        if id1 in result.values[i] :
            item = (test.iloc[i,0],test.iloc[i,1])
            first.append(item) 
    firstideal = pd.DataFrame(first, columns =['x', 'y'])

    # List all mapped test points for Second ideal function
    second = []
    for i in range(n):
        if id2 in result.values[i] :
            item = (test.iloc[i,0],test.iloc[i,1])
            second.append(item)
    secondideal = pd.DataFrame(second, columns =['x', 'y'])

    # List all mapped test points for Third ideal function
    third = []
    for i in range(n):
        if id3 in result.values[i] :
            item = (test.iloc[i,0],test.iloc[i,1])
            third.append(item)
    thirdideal = pd.DataFrame(third, columns =['x', 'y'])

    # List all mapped test points for Forth ideal function
    forth = []
    for i in range(n):
        if id4 in result.values[i] :
            item = (test.iloc[i,0],test.iloc[i,1])
            forth.append(item)
    forthideal = pd.DataFrame(forth, columns =['x', 'y'])
    
    ######################################################################################################
    # #                    Ploting graph Set 2                                                          ##
    #   Plot graph of mapped test points and ideal function
    #   You will see 4 graphs here

    x_map = [firstideal.x.tolist(),secondideal.x.tolist(),thirdideal.x.tolist(),forthideal.x.tolist()]
    y_map = [firstideal.y.tolist(),secondideal.y.tolist(),thirdideal.y.tolist(),forthideal.y.tolist()]
    best2=[]
    y_bestfit = []
    for i in range(4):  
        best2.append(best_ideal[i])
        y_bestfit.append(ideal[best2[i]].values) 

    for i in range(4):   
        plt.scatter(x_map[i],y_map[i])
        plt.plot(x_ideal,y_bestfit[i],c="red")
        plt.title("Ideal function {} and all mapped test points".format(i+1))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(["Mapped test points","Ideal function"])#, loc="upper right")
        plt.show()

    #######################################################################################################
    # Create The database table of the test-data, with mapping and y-deviation
    try:
        
        if conn.is_connected():
            # Creating table 3 for the test-data, with mapping and y-deviation in the database
            cursor.execute('DROP TABLE IF EXISTS Table3_Result;')
            cursor.execute("CREATE TABLE Table3_Result(`X (test func)` varchar(255),`Y (test func)` varchar(255),`Delta Y (test func)` varchar(255),`No. of ideal func` varchar(255))")
        
            # Inserted into Table3_Result in database")
            for i,row in df_result.iterrows():#loop through the data frame
                #here %S means string values 
                sql = "INSERT INTO ksassignment.Table3_Result VALUES (%s,%s,%s,%s)"
                cursor.execute(sql, tuple(row))
                # the connection is not auto committed by default, so we must commit to save our changes
                conn.commit()
            print("\nTable 3 Result is now created in the database.")
            print("This table contains a final result with four columns of x-values and y-values")
            print("as well as the corresponding chosen ideal function and the related deviation") 
            
    except Error as e:
                print("Error while connecting to MySQL", e)
    
    """####################################################################################
    # here is another code for creating Table3_Result
    # I am using above one because I want to try something different. The code below is just the same coding
    # to create the table as the one created above for Table 1 and Table 2 at the beginning.
    # You can use either of them. They both work perfectly.

    # Creating table 3 for the test-data, with mapping and y-deviation in the database
    Table3 = db.Table("Table3_Result", meta_train,
    db.Column("X (test func)", db.Float, nullable=False),
    db.Column("Y (test func)", db.Float, nullable=False),
    db.Column("Delta Y (test func)", db.Float, nullable=False),
    db.Column("No. of ideal func", db.String(5), nullable=False))
    meta_train.create_all(engine)

    # Delete existing data in Table 3 in the the database
    sql_query_delete_train = db.delete(Table3)
    result_delete_train = connection.execute(sql_query_delete_train)

    # Insert train Dataframe into Table 3 in the database
    Table3 = db.Table("Table3_Result",meta_train, autoload=True, autoload_with=engine)
    sql_query_train = db.insert(Table3)

    for row in df_result.itertuples():
        fill_table3 = [{
        "X (test func)":row.Xtest, "Y (test func)":row.Ytest,
        "Delta Y (test func)":row.DeltaY, "No. of ideal func":row.noideal}]
        result_load_train = connection.execute(sql_query_train, fill_table3)

    print("\nTable 3 Result is now created in the database.")
    print("This table contains a final result with four columns of x-values and y-values")
    print("as well as the corresponding chosen ideal function and the related deviation") 
    #####################################################################################"""
    """"""


if __name__ == '__main__':
    main()
    