import unittest
import numpy as np
from mos import devkit

class TestSin(unittest.TestCase):

    def test_contruction(self):

        x = devkit.model.variable.VariableScalar(name='x')
        f = devkit.model.sin(x)
        self.assertTrue(isinstance(f, devkit.model.sin))
        self.assertEqual(f.name, 'sin')
        self.assertEqual(len(f.arguments), 1)
        self.assertTrue(f.arguments[0] is x)

        self.assertRaises(TypeError, devkit.model.sin, [x, 1., 2.])

    def test_constant(self):

        a = devkit.model.constant.Constant(4.)
        
        f = devkit.model.sin(a)
        self.assertTrue(f.is_constant())
        self.assertEqual(f.get_value(), np.sin(4))

    def test_scalar(self):
        
        x = devkit.model.variable.VariableScalar(name='x', value=2.)

        f = devkit.model.sin(x)
        self.assertTrue(isinstance(f, devkit.model.sin))
        self.assertEqual(f.name, 'sin')
        self.assertEqual(len(f.arguments), 1)
        self.assertTrue(f.arguments[0] is x)
        
        self.assertEqual(f.get_value(), np.sin(2.))
        self.assertEqual(str(f), 'sin(x)')

    def test_matrix(self):

        value = np.random.randn(2,3)
        x = devkit.model.variable.VariableMatrix(name='x', value=value)

        f = devkit.model.sin(x)
        self.assertTrue(isinstance(f, devkit.model.expression.ExpressionMatrix))
        for i in range(2):
            for j in range(3):
                fij = f[i,j]
                self.assertTrue(isinstance(fij, devkit.model.sin))
                self.assertEqual(len(fij.arguments), 1)
                self.assertTrue(fij.arguments[0] is x[i,j])
                self.assertEqual(fij.get_value(), np.sin(value[i,j]))
                self.assertEqual(str(fij), 'sin(x[%d,%d])' %(i,j))

    def test_function(self):

        rn = devkit.model.utils.repr_number

        x = devkit.model.variable.VariableScalar(name='x', value=3.)

        f = devkit.model.sin(3*x)
        self.assertEqual(f.get_value(), np.sin(3.*3.))
        self.assertEqual(str(f), 'sin(x*%s)' %rn(3.))

        f = devkit.model.sin(x + 1)
        self.assertEqual(f.get_value(), np.sin(3.+1.))
        self.assertEqual(str(f), 'sin(x + %s)' %rn(1.))

        f = devkit.model.sin(devkit.model.sin(x))
        self.assertEqual(f.get_value(), np.sin(np.sin(3.)))
        self.assertEqual(str(f), 'sin(sin(x))')

    def test_analyze(self):

        x = devkit.model.variable.VariableScalar(name='x')
        y = devkit.model.variable.VariableScalar(name='y')

        f = devkit.model.sin(x)
        prop = f.__analyze__()
        self.assertFalse(prop['affine'])
        self.assertTrue(prop['b'] is np.NaN)
        self.assertEqual(len(prop['a']), 1.)
        self.assertTrue(x in prop['a'])

        f = devkit.model.sin(-4.*(-y+3*x-2))
        prop = f.__analyze__()
        self.assertFalse(prop['affine'])
        self.assertTrue(prop['b'] is np.NaN)
        self.assertEqual(len(prop['a']), 2.)
        self.assertTrue(x in prop['a'])
        self.assertTrue(y in prop['a'])

        f = devkit.model.sin((4.+x)*(-y+3*x-2))
        prop = f.__analyze__()
        self.assertFalse(prop['affine'])
        self.assertTrue(prop['b'] is np.NaN)
        self.assertEqual(len(prop['a']), 2.)
        self.assertTrue(x in prop['a'])
        self.assertTrue(y in prop['a'])

    def test_derivative(self):

        x = devkit.model.variable.VariableScalar(name='x', value=2.)
        y = devkit.model.variable.VariableScalar(name='y', value=3.)
        
        f = devkit.model.sin(x)
        fx = f.get_derivative(x)
        fy = f.get_derivative(y)

        self.assertTrue(isinstance(fx, devkit.model.function.Function))
        self.assertEqual(fx.get_value(), np.cos(2.))
        self.assertTrue(isinstance(fy, devkit.model.constant.Constant))
        self.assertEqual(fy.get_value(), 0.)

        f = devkit.model.sin(x + devkit.model.sin(x*y))
        fx = f.get_derivative(x)
        fy = f.get_derivative(y)

        self.assertTrue(fx.is_function())
        self.assertAlmostEqual(fx.get_value(), np.cos(2.+np.sin(2.*3.))*(1.+np.cos(2.*3.)*3.))

        self.assertTrue(fy.is_function())
        self.assertAlmostEqual(fy.get_value(), np.cos(2.+np.sin(2.*3.))*(0.+np.cos(2.*3.)*2.))

    def test_std_components(self):

        x = devkit.model.variable.VariableScalar('x', value=2.)
        y = devkit.model.variable.VariableScalar('y', value=3.)

        f = devkit.model.sin(x + y*x)

        comp = f.__get_std_components__()
        phi = comp['phi']
        gphi_list = comp['gphi_list']
        Hphi_list = comp['Hphi_list']
        
        self.assertTrue(phi is f)
        self.assertEqual(len(gphi_list), 2)

        v, exp = gphi_list[0]
        self.assertTrue(v is x)
        self.assertTrue(exp.is_function())
        self.assertEqual(exp.get_value(), np.cos(2.+2.*3.)*(1.+3.))
        
        v, exp = gphi_list[1]
        self.assertTrue(v is y)
        self.assertTrue(exp.is_function())
        self.assertEqual(exp.get_value(), np.cos(2.+2.*3.)*(0.+2.))

        self.assertEqual(len(Hphi_list), 3)

        # f sin(x + y*x)
        # fx = cos(x + y*x)*(1 + y) = cos(x + y*x) + cos(x + y*x)*y
        # fy = cos(x + y*x)*x
        # fxx = -sin(x + y*x)(1 + y) - sin(x + y*x)*y*(1 + y)
        # fxy = -sin(x + y*x)*x - sin(x + y*x)*y*x + cos(x + y*x)
        # fyy = -sin(x + y*x)*x*x
        
        v1, v2, exp = Hphi_list[0]
        self.assertTrue(v1 is x)
        self.assertTrue(v2 is x)
        self.assertAlmostEqual(exp.get_value(), -np.sin(2.+6.)*(1.+3.)-np.sin(2.+6.)*3*(1.+3.))

        v1, v2, exp = Hphi_list[1]
        self.assertTrue(v1 is x)
        self.assertTrue(v2 is y)
        self.assertAlmostEqual(exp.get_value(), -np.sin(2.+6.)*2-np.sin(2.+6.)*3.*2.+np.cos(2.+6.))

        v1, v2, exp = Hphi_list[2]
        self.assertTrue(v1 is y)
        self.assertTrue(v2 is y)
        self.assertAlmostEqual(exp.get_value(), -np.sin(2.+6.)*2*2)
        
