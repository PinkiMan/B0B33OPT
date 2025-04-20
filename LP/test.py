import unittest
import numpy as np

from lp import vyhra, vyhra2, minimaxfit


class MyTestCase(unittest.TestCase):
    def test_vyhra(self):
        c = np.array([1.27, 1.02, 4.70, 3.09, 9.00])
        k = 3000
        x = vyhra(c, k)
        x1 = np.array([3.40905816e-06, 2.69461077e+03, 9.28004584e-07, 1.68174891e-08, 3.05389222e+02])
        for item_1, item_2 in zip(x, x1):
            self.assertAlmostEqual(item_1, item_2, 4, f"{item_1} != {item_2} (diff = {abs(item_1-item_2)})")

    def test_vyhra2(self):
        c = np.array([1.27, 4.70, 9.00])
        k = 3000
        m = 400
        x = vyhra2(c, k, m)
        x1 = np.array([2046.90108498,  553.09881711,  400.00004477])
        for item_1, item_2 in zip(x, x1):
            self.assertAlmostEqual(item_1, item_2, 3, f"{item_1} != {item_2} (diff = {abs(item_1-item_2)})")

    def test_minimaxfit(self):
        x = np.array([[1, 2, 3, 3, 2], [4, 1, 2, 5, 6], [7, 8, 9, -5, 7]])
        y = np.array([[7, 4, 1, 2, 5]])
        a, b, r = minimaxfit(x, y)

        a1 = [-2.776, 0.194, -0.030]
        b1 = 9.403
        r1 = 0.194

        for item_1, item_2 in zip(a, a1):
            self.assertAlmostEqual(item_1, item_2, 2, f"{item_1} != {item_2}")
        self.assertAlmostEqual(b, b1, 4)
        self.assertAlmostEqual(r, r1, 4)

if __name__ == '__main__':
    unittest.main()
