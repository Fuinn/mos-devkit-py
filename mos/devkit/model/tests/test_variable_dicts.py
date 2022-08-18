import unittest
import numpy as np
from mos import devkit

class TestVariableDicts(unittest.TestCase):

    def test_construction_with_tuples(self):

        x = devkit.model.VariableDict([(1,2), ('tt', 4)], name='x')

        self.assertTrue(isinstance(x[(1,2)], devkit.model.VariableScalar))
        self.assertTrue(isinstance(x[('tt',4)], devkit.model.VariableScalar))

        self.assertRaises(KeyError, lambda a: x[a], 50)
        
    def test_construction(self):

        x = devkit.model.VariableDict(['a', 'b'], name='foo')

        self.assertTrue(isinstance(x, dict))

        self.assertEqual(len(x), 2)
        
        xa = x['a']
        self.assertTrue(isinstance(xa, devkit.model.VariableScalar))
        self.assertEqual(xa.get_value(), 0.)
        self.assertTrue(xa.is_continuous())
        self.assertEqual(xa.name, 'foo_a')

        xb = x['b']
        self.assertTrue(isinstance(xb, devkit.model.VariableScalar))
        self.assertEqual(xb.get_value(), 0.)
        self.assertTrue(xb.is_continuous())
        self.assertEqual(xb.name, 'foo_b')

        x = devkit.model.VariableDict(['a', 'b'], name='bar', value={'a': 10, 'c': 100})

        self.assertTrue(isinstance(x, dict))

        self.assertEqual(len(x), 2)
        
        xa = x['a']
        self.assertTrue(isinstance(xa, devkit.model.VariableScalar))
        self.assertEqual(xa.get_value(), 10.)
        self.assertTrue(xa.is_continuous())
        self.assertEqual(xa.name, 'bar_a')

        xb = x['b']
        self.assertTrue(isinstance(xb, devkit.model.VariableScalar))
        self.assertEqual(xb.get_value(), 0.)
        self.assertTrue(xb.is_continuous())
        self.assertEqual(xb.name, 'bar_b')
        

        
