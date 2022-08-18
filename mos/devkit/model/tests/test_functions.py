import unittest
import numpy as np
from mos import devkit

class TestFunctions(unittest.TestCase):

    def test_contruction(self):

        x = devkit.model.variable.VariableScalar(name='x')
        
        f = devkit.model.function.Function([x, devkit.model.expression.make_Expression(1.)])

        self.assertTrue(isinstance(f, devkit.model.function.Function))
        self.assertEqual(len(f.arguments), 2)
        self.assertTrue(f.arguments[0] is x)
        self.assertTrue(isinstance(f.arguments[1], devkit.model.constant.Constant))
        self.assertEqual(f.arguments[1].get_value(), 1.)

    def test_get_derivative(self):

        pass

    def test_is_type(self):
        
        f = devkit.model.function.Function()
        self.assertFalse(f.is_constant())
        self.assertFalse(f.is_variable())
        self.assertTrue(f.is_function())
