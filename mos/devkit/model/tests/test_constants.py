import unittest
import numpy as np
from mos import devkit

class TestConstants(unittest.TestCase):

    def test_construction(self):

        c = devkit.model.constant.Constant(4.)
        self.assertTrue(isinstance(c, devkit.model.expression.Expression))
        self.assertTrue(isinstance(c, devkit.model.constant.Constant))
        self.assertEqual(c.name, 'const')
        self.assertEqual(c.get_value(), 4.)

        self.assertRaises(TypeError, devkit.model.constant.Constant, [1,2,3])
        self.assertRaises(TypeError, devkit.model.constant.Constant, devkit.model.constant.Constant(3.))

    def test_get_variables(self):

        c = devkit.model.constant.Constant(4.)

        self.assertSetEqual(c.get_variables(), set())

    def test_repr(self):

        c = devkit.model.constant.Constant(5.)
        s = str(c)
        self.assertEqual(s, devkit.model.utils.repr_number(5.))

    def test_value(self):

        c = devkit.model.constant.Constant(6.)
        self.assertEqual(c.get_value(), 6.)        

    def test_is_zero(self):

        c = devkit.model.constant.Constant(2.)
        self.assertFalse(c.is_zero())
        c = devkit.model.constant.Constant(0.)
        self.assertTrue(c.is_zero())

    def test_is_one(self):

        c = devkit.model.constant.Constant(2.)
        self.assertFalse(c.is_one())
        c = devkit.model.constant.Constant(1.)
        self.assertTrue(c.is_one())

    def test_is_type(self):

        c = devkit.model.constant.Constant(4.)
        self.assertTrue(c.is_constant())
        self.assertFalse(c.is_variable())
        self.assertFalse(c.is_function())
        
        self.assertFalse(c.is_constant(5.))
        self.assertTrue(c.is_constant(4.))

    def test_derivatives(self):

        c = devkit.model.constant.Constant(4.)
        x = devkit.model.variable.VariableScalar('x')

        dc = c.get_derivative(x)
        self.assertTrue(dc.is_constant())
        self.assertEqual(dc.get_value(), 0.)

    def test_std_components(self):

        c = devkit.model.constant.Constant(4.)

        comp = c.__get_std_components__()
        phi = comp['phi']
        gphi_list = comp['gphi_list']
        Hphi_list = comp['Hphi_list']

        self.assertTrue(phi is c)
        self.assertEqual(len(gphi_list), 0)
        self.assertEqual(len(Hphi_list), 0)
