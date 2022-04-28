import unittest
from KSpythonAssignment import calculations 

# Creat unit-tests for useful elements
class UnitTestforcalculation(unittest.TestCase):

    def test_leastsqaure(self):  
        #test least square
        cal = calculations()
        result = cal.leastsquare(9, 2)
        self.assertEqual(result, 49, "The result of least square should be 49")
    
    def test_subtraction(self):
        #test subtraction
        cal = calculations()
        result = cal.subtraction(7, 4)
        self.assertEqual(result, 3, "The subtraction should be 3")

    def test_MSEdev(self):
        #test Mean Squared Error
        cal = calculations()
        result = cal.MSEdev(20, 12)
        self.assertEqual(result, 0.64, "The result of Mean Squared Error should be 0.64")

    def test_RMSEdev(self):
        #test Root Mean Squared Error
        cal = calculations()
        result = cal.RMSEdev(20, 12)
        self.assertEqual(result, 0.8, "The result of Root Mean Squared Error should be 0.8")

    def test_MAEdev(self):
        #test Mean Absolute Error
        cal = calculations()
        result = cal.MAEdev(70, 40)
        self.assertEqual(result, 0.3, "The result of Mean Absolute Error should be 0.3")
    
if __name__ == "__main__":
    unittest.main()