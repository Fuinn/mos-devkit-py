import unittest
import numpy as np
from numpy.linalg import norm
from mos import devkit
from mos.devkit.model import minimize, maximize, EmptyObjective, cos, sin

class TestModels(unittest.TestCase):

    def test_var_order_consistency(self):
        
        x1 = devkit.model.variable.VariableScalar('x1')
        x2 = devkit.model.variable.VariableScalar('x2')
        x3 = devkit.model.variable.VariableScalar('x3')
        x4 = devkit.model.variable.VariableScalar('x4')
        f = x1 + x2
        constraints= [x1 >= x2, x2 + x3 <= -4, 4*x4 >= x1]

        p = devkit.model.Model(minimize(f), constraints)

        std_prob = p.__get_std_problem__()

        A0 = std_prob.A.copy()

        for i in range(1000):
            x1 = devkit.model.variable.VariableScalar('x1')
            x2 = devkit.model.variable.VariableScalar('x2')
            x3 = devkit.model.variable.VariableScalar('x3')
            x4 = devkit.model.variable.VariableScalar('x4')
            f = x1 + x2
            constraints= [x1 >= x2, x2 + x3 <= -4, 4*x4 >= x1]
            p = devkit.model.Model(minimize(f), constraints)
            A1 = p.__get_std_problem__().A
            self.assertTrue(np.all(A0.row == A1.row))
            self.assertTrue(np.all(A0.col == A1.col))
            self.assertTrue(np.all(A0.data == A1.data))
            
    def test_construction(self):
        
        x = devkit.model.variable.VariableScalar('x')

        f = x*x

        c1 = x >= 1
        c2 = x <= 5
        c = [c1, c2]

        p = devkit.model.Model(minimize(f), c)

        self.assertTrue(isinstance(p.objective, minimize))
        self.assertTrue(p.objective.get_function() is f)

        self.assertEqual(len(p.constraints), 2)
        self.assertTrue(p.constraints[0] is c1)
        self.assertTrue(p.constraints[1] is c2)

        p = devkit.model.Model()
        self.assertTrue(p.objective.get_function().is_constant(0.))
        self.assertEqual(len(p.constraints), 0)

        p = devkit.model.Model(maximize(f))
        self.assertTrue(isinstance(p.objective, maximize))
        self.assertTrue(p.objective.get_function() is f)
        self.assertEqual(len(p.constraints), 0)

        p = devkit.model.Model()
        self.assertEqual(len(p.constraints), 0)
        self.assertTrue(isinstance(p.objective, devkit.model.model.EmptyObjective))

        p = devkit.model.Model(objective=None)
        self.assertEqual(len(p.constraints), 0)
        self.assertTrue(isinstance(p.objective, devkit.model.model.EmptyObjective))

        self.assertRaises(TypeError, devkit.model.Model, objective=f)
        self.assertRaises(TypeError, devkit.model.Model, None, ['foo'])

    def test_construction_constr_not_flat(self):

        x = devkit.model.variable.VariableMatrix('x', value=np.random.randn(3,2))
        y = devkit.model.variable.VariableScalar('y')

        c0 = x == 0
        self.assertTrue(isinstance(c0, devkit.model.constraint.ConstraintArray))

        c1 = y == 2

        p = devkit.model.Model(EmptyObjective(), constraints=[c0, [[c1]]])
        
        self.assertEqual(len(p.constraints), 7)
        for c in p.constraints:
            self.assertTrue(isinstance(c, devkit.model.constraint.Constraint))

    def test_std_components(self):

        x = devkit.model.variable.VariableScalar('x')
        y = devkit.model.variable.VariableScalar('y')

        f = x*x + 2*x*y + y*y
        c0 = x <= 10
        c1 = x + y == 3
        c2 = 3*x <= 10
        c3 = y >= 19.
        c4 = x*y == 20.
        c5 = devkit.model.sin(x) <= 3
        c6 = devkit.model.cos(y) >= 4.

        p = devkit.model.Model(minimize(f), [c0, c1, c2, c3, c4, c5, c6])
        
        comp = p.__get_std_components__()

        self.assertEqual(len(comp), 14)

        phi = comp['phi']
        gphi_list = comp['gphi_list']
        Hphi_list = comp['Hphi_list']
        phi_prop = comp['phi_prop']

        cA_list = comp['cA_list']
        cJ_list = comp['cJ_list']
        A_list = comp['A_list']
        b_list = comp['b_list']
        f_list = comp['f_list']
        J_list = comp['J_list']
        H_list = comp['H_list']
        u_list = comp['u_list']
        l_list = comp['l_list']
        prop_list = comp['prop_list']

        self.assertEqual(len(cA_list), len(b_list))
        self.assertEqual(len(cJ_list), len(f_list))

        self.assertEqual(str(phi), 'x*x + x*2.00e+00*y + y*y')
        self.assertEqual(str(gphi_list), '[(x, x + x + y*2.00e+00), (y, x*2.00e+00 + y + y)]')
        self.assertEqual(str(Hphi_list), '[(x, x, 2.00e+00), (x, y, 2.00e+00), (y, y, 2.00e+00)]')

        self.assertEqual(str(A_list), '[(0, x, 1.0), (0, y, 1.0), (1, x, 3.0), (1, s, -1.0)]')
        self.assertEqual(str(b_list), '[3.0, 10.0]')

        self.assertEqual(str(f_list), '[x*y + -2.00e+01, sin(x) + -3.00e+00 + s*-1.00e+00, cos(y) + -4.00e+00 + s*-1.00e+00]')
        self.assertEqual(str(J_list), '[(0, x, y), (0, y, x), (1, x, cos(x)), (1, s, -1.00e+00), (2, y, sin(y)*-1.00e+00), (2, s, -1.00e+00)]')
        self.assertEqual(str(H_list), '[[(x, y, 1.00e+00)], [(x, x, sin(x)*-1.00e+00)], [(y, y, -1.00e+00*cos(y))]]')

        self.assertEqual(str([(x[0], x[1]) for x in u_list]), '[(x, 10.0), (s, 0), (s, 0)]')
        self.assertEqual(str([(x[0], x[1]) for x in l_list]), '[(y, 19.0), (s, 0)]')

    def test_std_problem(self):

        np.random.seed(0)

        x = devkit.model.variable.VariableScalar('x', value=3.)
        y = devkit.model.variable.VariableScalar('y', value=4.)
        
        f = x*x + 2*x*y + y*y
        c0 = x <= 10
        c1 = x + y == 3
        c2 = 3*x <= 10           # slack s1
        c3 = y >= 19.
        c4 = x*y == 20.
        c5 = devkit.model.sin(x) <= 3  # slack s2
        c6 = devkit.model.cos(y) >= 4. # slack s3

        p = devkit.model.Model(minimize(f), [c0, c1, c2, c3, c4, c5, c6])

        std_prob = p.__get_std_problem__()
        
        # vars
        self.assertEqual(len(std_prob.var2index), 5)
        self.assertEqual(len(std_prob.index2var), 5)
        index_x = std_prob.var2index[x]
        index_y = std_prob.var2index[y]
        index_s1 = std_prob.var2index[c2.slack]
        index_s2 = std_prob.var2index[c5.slack]
        index_s3 = std_prob.var2index[c6.slack]

        # x
        self.assertEqual(std_prob.x.size, 5)
        temp = np.zeros(5)
        temp[index_x] = 3.
        temp[index_y] = 4.
        self.assertTrue(np.all(std_prob.x == temp))

        # A, b
        self.assertTupleEqual(std_prob.A.shape, (2, 5))
        self.assertEqual(std_prob.A.nnz, 4)
        self.assertTrue(np.all(std_prob.A.row == np.array([0, 0, 1, 1])))
        self.assertTrue(np.all(std_prob.A.col == np.array([index_x, index_y, index_x, index_s1])))
        self.assertTrue(np.all(std_prob.A.data == np.array([1, 1, 3, -1])))
        self.assertTrue(np.all(std_prob.b == np.array([3., 10.])))

        # u, l
        inf = 1e8
        self.assertEqual(std_prob.u.size, 5)
        self.assertEqual(std_prob.l.size, 5)
        temp = np.zeros(5)
        temp[index_x] = 10.
        temp[index_y] = inf
        temp[index_s1] = 0
        temp[index_s2] = 0
        temp[index_s3] = inf
        self.assertTrue(np.all(std_prob.u == temp))
        temp = np.zeros(5)
        temp[index_x] = -inf
        temp[index_y] = 19
        temp[index_s1] = -inf
        temp[index_s2] = -inf
        temp[index_s3] = 0
        self.assertTrue(np.all(std_prob.l == temp))

        # integer flags
        self.assertTrue(isinstance(std_prob.P, np.ndarray))
        self.assertEqual(std_prob.P.size, 5)
        self.assertNotEqual(std_prob.P.dtype, int)
        self.assertEqual(std_prob.P.dtype, bool)
        self.assertTrue(np.all(std_prob.P == False))
        self.assertFalse(np.any(std_prob.P == True))

        # phi, gphi, Hphi before eval
        self.assertEqual(std_prob.phi, 0.)
        self.assertTrue(np.all(std_prob.gphi == np.zeros(5)))
        self.assertTupleEqual(std_prob.Hphi.shape, (5, 5))
        self.assertEqual(std_prob.Hphi.nnz, 3)
        if index_x <= index_y:
            self.assertTrue(np.all(std_prob.Hphi.row == np.array([index_x, index_y, index_y])))
            self.assertTrue(np.all(std_prob.Hphi.col == np.array([index_x, index_x, index_y])))
        else:
            self.assertTrue(np.all(std_prob.Hphi.row == np.array([index_x, index_x, index_y])))
            self.assertTrue(np.all(std_prob.Hphi.col == np.array([index_x, index_y, index_y])))
        self.assertTrue(np.all(std_prob.Hphi.data == np.zeros(3)))

        # f, J, H_combined before eval
        self.assertEqual(std_prob.f.size, 3)
        self.assertTrue(np.all(std_prob.f == np.zeros(3)))
        self.assertTupleEqual(std_prob.J.shape, (3, 5))
        self.assertEqual(std_prob.J.nnz, 6)
        self.assertTrue(np.all(std_prob.J.row == np.array([0, 0, 1, 1, 2, 2])))
        self.assertTrue(np.all(std_prob.J.col == np.array([index_x, index_y, index_x, index_s2, index_y, index_s3])))
        self.assertTrue(np.all(std_prob.J.data == np.zeros(6)))
        self.assertEqual(std_prob.H_combined.nnz, 3)
        if index_x <= index_y:
            self.assertTrue(np.all(std_prob.H_combined.row == np.array([index_y, index_x, index_y])))
            self.assertTrue(np.all(std_prob.H_combined.col == np.array([index_x, index_x, index_y])))
        else:
            self.assertTrue(np.all(std_prob.H_combined.row == np.array([index_x, index_x, index_y])))
            self.assertTrue(np.all(std_prob.H_combined.col == np.array([index_y, index_x, index_y])))
        self.assertTrue(np.all(std_prob.H_combined.data == np.zeros(3)))
        
        var = np.random.randn(5)

        # Eval
        std_prob.eval(var)

        # phi, gphi, Hphi after eval
        self.assertAlmostEqual(std_prob.phi, (var[index_x]*var[index_x] +
                                              2.*var[index_x]*var[index_y] +
                                              var[index_y]*var[index_y]))
        temp = np.zeros(5)
        temp[index_x] = 2.*var[index_x] + 2.*var[index_y]
        temp[index_y] = 2.*var[index_x] + 2.*var[index_y]
        self.assertTrue(norm(std_prob.gphi - temp) < 1e-8)
        self.assertTrue(np.all(std_prob.Hphi.data == np.array([2., 2., 2.])))

        # f, J after eval
        self.assertTrue(norm(std_prob.f - np.array([var[index_x]*var[index_y]-20.,
                                                    np.sin(var[index_x])-var[index_s2]-3.,
                                                    np.cos(var[index_y])-var[index_s3]-4.])) < 1e-8)
        self.assertTrue(np.all(std_prob.J.data == np.array([var[index_y],
                                                            var[index_x],
                                                            np.cos(var[index_x]),
                                                            -1.,
                                                            -np.sin(var[index_y]),
                                                            -1.])))

        # Combine H - ones
        std_prob.combine_H(np.ones(3))
        self.assertTrue(np.all(std_prob.H_combined.data == np.array([1.,
                                                                     -np.sin(var[index_x]),
                                                                     -np.cos(var[index_y])])))

        # Combine H - rand
        lam = np.random.randn(3)
        std_prob.combine_H(lam)
        self.assertTrue(np.all(std_prob.H_combined.data == np.array([1.*lam[0],
                                                                     -np.sin(var[index_x])*lam[1],
                                                                     -np.cos(var[index_y])*lam[2]])))

        # Properties
        self.assertTrue(len(std_prob.properties), 3)
        self.assertTrue('continuous' in std_prob.properties)
        self.assertTrue('optimization' in std_prob.properties)
        self.assertTrue('nonlinear' in std_prob.properties)        

    def test_solve_LP(self):

        x = devkit.model.VariableScalar('x')
        y = devkit.model.VariableScalar('y')

        # Model
        p = devkit.model.Model(objective=maximize(-2*x+5*y),
                         constraints=[100 <= x,
                                      x <= 200,
                                      80 <= y,
                                      y <= 170,
                                      y == -x + 200])

        # std prob
        std_prob = p.__get_std_problem__()
        self.assertListEqual(std_prob.properties, ['linear', 'continuous', 'optimization'])
                
        info = p.solve(parameters={'quiet': True})

        self.assertTrue('iterations' in info)
        self.assertTrue('time_transformation' in info)
        self.assertTrue('time_solver' in info)

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 100, places=4)
        self.assertAlmostEqual(y.get_value(), 100, places=4)

        info = p.solve(solver=devkit.solver.SolverAugL(), parameters={'quiet': True})

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 100, places=4)
        self.assertAlmostEqual(y.get_value(), 100, places=4)

    def test_solve_LP_clp(self):

        x = devkit.model.VariableScalar('x')
        y = devkit.model.VariableScalar('y')

        # Model
        p = devkit.model.Model(objective=maximize(-2*x+5*y),
                         constraints=[100 <= x,
                                      x <= 200,
                                      80 <= y,
                                      y <= 170,
                                      y >= -x + 200])

        try:
            info = p.solve(solver=devkit.solver.SolverClp(), parameters={'quiet': True})
        except ImportError:
            raise unittest.SkipTest('clp not available')

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 100, places=4)
        self.assertAlmostEqual(y.get_value(), 170, places=4)

    def test_solve_LP_clp_cmd(self):

        x = devkit.model.VariableScalar('x')
        y = devkit.model.VariableScalar('y')

        # Model
        p = devkit.model.Model(objective=maximize(-2*x+5*y),
                         constraints=[100 <= x,
                                      x <= 200,
                                      80 <= y,
                                      y <= 170,
                                      y >= -x + 200])

        try:
            info = p.solve(solver=devkit.solver.SolverClpCMD(), parameters={'quiet': True})
        except ImportError:
            raise unittest.SkipTest('clp cmd not available')

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 100, places=4)
        self.assertAlmostEqual(y.get_value(), 170, places=4)

    def test_solve_LP_clp_cmd_duals_ub(self):

        x = devkit.model.VariableScalar('x')
        y = devkit.model.VariableScalar('y')

        c1 = 100 <= x
        c2 = x <= 200
        c3 = 80 <= y
        c4 = y <= 170
        c5 = y == -x + 200

        # Model
        p = devkit.model.Model(objective=minimize(2*x-5*y),
                         constraints=[c1, c2, c3, c4, c5])

        try:
            info = p.solve(solver=devkit.solver.SolverClpCMD(), parameters={'quiet': True})
        except ImportError:
            raise unittest.SkipTest('clp cmd not available')

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 100, places=4)
        self.assertAlmostEqual(y.get_value(), 100, places=4)

        self.assertEqual(c2.get_dual(), 0.)
        self.assertEqual(c3.get_dual(), 0.)
        self.assertEqual(c4.get_dual(), 0.)

        self.assertEqual(c1.get_dual(), 7.)
        self.assertEqual(c5.get_dual(), -5.)

    def test_solve_LP_clp_cmd_duals_A(self):

        x = devkit.model.VariableScalar('x')

        c = 3*x == 4

        p = devkit.model.Model(objective=minimize(5*x),
                         constraints=[c])

        try:
            info = p.solve(solver=devkit.solver.SolverClpCMD(), parameters={'quiet': True})
        except ImportError:
            raise unittest.SkipTest('clp cmd not available')

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 4./3., places=4)

        self.assertLess(np.abs(5.-3*c.get_dual()), 1e-7)
        
    def test_solve_LP_clp_cmd_duals_A_ineq(self):

        x = devkit.model.VariableScalar('x')

        c = 3*x >= 4 # -3x <= -4

        p = devkit.model.Model(objective=minimize(5*x),
                         constraints=[c])

        try:
            info = p.solve(solver=devkit.solver.SolverClpCMD(), parameters={'quiet': True})
        except ImportError:
            raise unittest.SkipTest('clp cmd not available')

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 4./3., places=4)

        self.assertLess(np.abs(5.-3*c.get_dual()), 1e-7)

        c = 3*x <= 4 

        p = devkit.model.Model(objective=minimize(-5*x),
                         constraints=[c])

        try:
            info = p.solve(solver=devkit.solver.SolverClpCMD(), parameters={'quiet': True})
        except ImportError:
            raise unittest.SkipTest('clp cmd not available')

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 4./3., places=4)

        self.assertLess(np.abs(-5.+3*c.get_dual()), 1e-7)

    def test_solve_ipopt_duals_J(self):

        x = devkit.model.VariableScalar('x')

        c1 = devkit.model.cos(x) == 0.75
        c2 = x >= -np.pi
        c3 = x <= np.pi

        p = devkit.model.Model(objective=minimize(5*x),
                         constraints=[c1, c2, c3])

        try:
            info = p.solve(solver=devkit.solver.SolverIpopt(), parameters={'quiet': True})
        except ImportError:
            raise unittest.SkipTest('ipopt not available')

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), -np.abs(np.arccos(0.75)), places=5)
        self.assertAlmostEqual(np.cos(x.get_value()), 0.75, places=5)
        self.assertGreater(x.get_value(), -np.pi)
        self.assertLess(x.get_value(), np.pi)

        self.assertEqual(c2.get_dual(), 0.)
        self.assertEqual(c3.get_dual(), 0.)

        self.assertLess(np.abs(5. - (-np.sin(x.get_value()))*c1.get_dual()), 1e-7)

    def test_solve_QP(self):

        x = devkit.model.VariableScalar('x')
        y = devkit.model.VariableScalar('y')

        f = 3*x*x+y*y+2*x*y+x+6*y+2

        # Model
        p = devkit.model.Model(objective=minimize(f))

        # std prob
        std_prob = p.__get_std_problem__()
        self.assertListEqual(std_prob.properties, ['nonlinear', 'continuous', 'optimization'])
                
        info = p.solve(solver=devkit.solver.SolverINLP(), parameters={'quiet': True})
        
        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 1.25, places=4)
        self.assertAlmostEqual(y.get_value(), -4.25, places=4)

        info = p.solve(solver=devkit.solver.SolverAugL(), parameters={'quiet': True})
        
        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 1.25, places=4)
        self.assertAlmostEqual(y.get_value(), -4.25, places=4)

        # Model
        p = devkit.model.Model(objective=minimize(f),
                         constraints=[2*x+3*y >= 4,
                                      x >= 0,
                                      y >= 0])

        info = p.solve(solver=devkit.solver.SolverINLP(), parameters={'quiet': True})

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 0.5, places=4)
        self.assertAlmostEqual(y.get_value(), 1, places=4)
        self.assertAlmostEqual(f.get_value(), 11.25, places=4)

        info = p.solve(solver=devkit.solver.SolverAugL(), parameters={'quiet': True})

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 0.5, places=4)
        self.assertAlmostEqual(y.get_value(), 1, places=4)
        self.assertAlmostEqual(f.get_value(), 11.25, places=4)

    def test_solve_NLP_unconstrained(self):

        n = 5

        x = devkit.model.VariableMatrix(name='x', value=[1.3, 0.7, 0.8, 1.9, 1.2], shape=(5,1))

        f = 0.
        for i in range(0, n-1):
            f = f + 100*(x[i+1,0]-x[i,0]*x[i,0])*(x[i+1,0]-x[i,0]*x[i,0]) + (1.-x[i,0])*(1.-x[i,0])

        # Model
        p = devkit.model.Model(minimize(f))

        # std prob
        std_prob = p.__get_std_problem__()
        self.assertListEqual(std_prob.properties, ['nonlinear', 'continuous', 'optimization'])

        try:
            info = p.solve(solver=devkit.solver.SolverIpopt(), parameters={'quiet': True}, fast_evaluator=True)
        except ImportError:
            raise unittest.SkipTest('ipopt not available')
        
        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(f.get_value(), 0, places=4)
        self.assertLess(norm(x.get_value()-np.ones((5,1)), np.inf), 1e-2)

        x.set_value(np.matrix([1.3, 0.7, 0.8, 1.9, 1.2]).T)

        info = p.solve(solver=devkit.solver.SolverIpopt(), parameters={'quiet': True}, fast_evaluator=False)

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(f.get_value(), 0, places=4)
        self.assertLess(norm(x.get_value()-np.ones((5,1)), np.inf), 1e-2)
        
    def test_solve_NLP_constrained(self):

        # Hock-Schittkowski
        # Problem 71
        
        x1 = devkit.model.VariableScalar('x1', value=1)
        x2 = devkit.model.VariableScalar('x2', value=5)
        x3 = devkit.model.VariableScalar('x3', value=5)
        x4 = devkit.model.VariableScalar('x4', value=1)

        f = x1*x4*(x1+x2+x3) + x3
        
        constraints = [x1*x2*x3*x4 >= 25,
                       x1*x1 + x2*x2 + x3*x3 + x4*x4 == 40,
                       1 <= x1, x1 <= 5,
                       1 <= x2, x2 <= 5,
                       1 <= x3, x3 <= 5,
                       1 <= x4, x4 <= 5]

        p = devkit.model.Model(minimize(f), constraints=constraints)

        # std prob
        std_prob = p.__get_std_problem__()
        self.assertListEqual(std_prob.properties, ['nonlinear', 'continuous', 'optimization'])

        try:
            info = p.solve(solver=devkit.solver.SolverIpopt(), parameters={'quiet': True}, fast_evaluator=True)
        except ImportError:
            raise unittest.SkipTest('ipopt not available')

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(f.get_value(), 17.0140173, places=3)
        self.assertAlmostEqual(x1.get_value(), 1., places=3)
        self.assertAlmostEqual(x2.get_value(), 4.7429994, places=3)
        self.assertAlmostEqual(x3.get_value(), 3.8211503, places=3)
        self.assertAlmostEqual(x4.get_value(), 1.3794082, places=3)
        
        x1.set_value(1.)
        x2.set_value(5.)
        x3.set_value(5.)
        x4.set_value(1.)

        info = p.solve(solver=devkit.solver.SolverIpopt(), parameters={'quiet': True}, fast_evaluator=False)
    
        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(f.get_value(), 17.0140173, places=3)
        self.assertAlmostEqual(x1.get_value(), 1., places=3)
        self.assertAlmostEqual(x2.get_value(), 4.7429994, places=3)
        self.assertAlmostEqual(x3.get_value(), 3.8211503, places=3)
        self.assertAlmostEqual(x4.get_value(), 1.3794082, places=3)

    def test_solve_MILP_cbc(self):

        x1 = devkit.model.VariableScalar('x1', type='integer')
        x2 = devkit.model.VariableScalar('x2', type='integer')
        x3 = devkit.model.VariableScalar('x3')
        x4 = devkit.model.VariableScalar('x4')

        obj = -x1-x2
        constr = [-2*x1+2*x2+x3 == 1,
                  -8*x1+10*x2+x4 == 13,
                  x4 >= 0,
                  x3 <= 0]
        
        p = devkit.model.Model(minimize(obj), constr)

        # std prob
        std_prob = p.__get_std_problem__()
        self.assertListEqual(std_prob.properties, ['linear', 'integer', 'optimization'])

        try:
            self.assertRaises(TypeError, p.solve, devkit.solver.SolverIpopt(), {'quiet': True})
        except ImportError:
            raise unittest.SkipTest('ipopt not available')
        self.assertRaises(TypeError, p.solve, devkit.solver.SolverAugL(), {'quiet': True})
        try:
            self.assertRaises(TypeError, p.solve, devkit.solver.SolverClp(), {'quiet': True})
        except ImportError:
            raise unittest.SkipTest('clp not available')
        self.assertRaises(TypeError, p.solve, devkit.solver.SolverINLP(), {'quiet': True})

        try:
            info = p.solve(devkit.solver.SolverCbc(), parameters={'quiet': True})
        except ImportError:
            raise unittest.SkipTest('cbc not available')

        self.assertEqual(info['status'], 'solved')
        self.assertEqual(x1.get_value(), 1.)
        self.assertEqual(x2.get_value(), 2.)

        x1.type = 'continuous'
        x2.type = 'continuous'

        try:
            info = p.solve(devkit.solver.SolverCbc(), parameters={'quiet': True})
        except ImportError:
            raise unittest.SkipTest('cbc not available')
        
        self.assertEqual(info['status'], 'solved')
        self.assertEqual(x1.get_value(), 4.)
        self.assertEqual(x2.get_value(), 4.5)

    def test_solve_MILP_cbc_cmd(self):

        x1 = devkit.model.VariableScalar('x1', type='integer')
        x2 = devkit.model.VariableScalar('x2', type='integer')
        x3 = devkit.model.VariableScalar('x3')
        x4 = devkit.model.VariableScalar('x4')

        obj = -x1-x2
        constr = [-2*x1+2*x2+x3 == 1,
                  -8*x1+10*x2+x4 == 13,
                  x4 >= 0,
                  x3 <= 0]
        
        p = devkit.model.Model(minimize(obj), constr)

        try:
            info = p.solve(devkit.solver.SolverCbcCMD(), parameters={'quiet': True})
        except ImportError:
            raise unittest.SkipTest('cbc cmd not available')

        self.assertEqual(info['status'], 'solved')
        self.assertEqual(x1.get_value(), 1.)
        self.assertEqual(x2.get_value(), 2.)

    def test_infeasible_MILP_cbc_cmd(self):

        x1 = devkit.model.VariableScalar('x1', type='integer')
        x2 = devkit.model.VariableScalar('x2', type='integer')
        x3 = devkit.model.VariableScalar('x3')
        x4 = devkit.model.VariableScalar('x4')

        obj = -x1-x2
        constr = [-2*x1+2*x2+x3 == 1,
                  -8*x1+10*x2+x4 == 13,
                  x1 >= 2,
                  x1 <= 1,
                  x4 >= 0,
                  x3 <= 0]
        
        p = devkit.model.Model(minimize(obj), constr)

        try:
            info = p.solve(devkit.solver.SolverCbcCMD(), parameters={'quiet': True})
        except ImportError:
            raise unittest.SkipTest('cbc cmd not available')

        self.assertEqual(info['status'], 'error')

    def test_infeasible_MILP_cplex_cmd(self):

        x1 = devkit.model.VariableScalar('x1', type='integer')
        x2 = devkit.model.VariableScalar('x2', type='integer')
        x3 = devkit.model.VariableScalar('x3')
        x4 = devkit.model.VariableScalar('x4')

        obj = -x1-x2
        constr = [-2*x1+2*x2+x3 == 1,
                  -8*x1+10*x2+x4 == 13,
                  x1 >= 2,
                  x1 <= 1,
                  x4 >= 0,
                  x3 <= 0]
        
        p = devkit.model.Model(minimize(obj), constr)

        try:
            info = p.solve(devkit.solver.SolverCplexCMD(), parameters={'quiet': True})
        except ImportError:
            raise unittest.SkipTest('cplex cmd not available')

        self.assertEqual(info['status'], 'error')
        
    def test_solve_feasibility(self):

        x = devkit.model.VariableScalar('x', value=1.)

        constr = [x*devkit.model.cos(x)-x*x == 0]
        
        p = devkit.model.Model(EmptyObjective(), constr)

        # std prob
        std_prob = p.__get_std_problem__()
        self.assertListEqual(std_prob.properties, ['nonlinear', 'continuous', 'feasibility'])

        try:
            self.assertRaises(TypeError, p.solve, devkit.solver.SolverClp(), {'quiet': True})
        except ImportError:
            raise unittest.SkipTest('clp not available')
        try:
            self.assertRaises(TypeError, p.solve, devkit.solver.SolverCbc(), {'quiet': True})
        except ImportError:
            raise unittest.SkipTest('cbc not available')

        info = p.solve(devkit.solver.SolverNR(), parameters={'quiet': True, 'feastol': 1e-10})
        
        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 0.739085133215161, places=7)

        x.set_value(1.)
        self.assertNotAlmostEqual(x.get_value(), 0.739085133215161, places=7)

        info = p.solve(devkit.solver.SolverNR(), parameters={'quiet': True, 'feastol': 1e-10}, fast_evaluator=False)

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(x.get_value(), 0.739085133215161, places=7)

    def test_solve_NLP_beam(self):

        N = 500
        h = 1./N
        alpha = 350.
        
        t = devkit.model.VariableMatrix('t', shape=(N+1,1))
        x = devkit.model.VariableMatrix('x', shape=(N+1,1))
        u = devkit.model.VariableMatrix('u', shape=(N+1,1))
        
        f = sum([0.5*h*(u[i,0]*u[i,0]+u[i+1,0]*u[i+1,0]) +
                 0.5*alpha*h*(cos(t[i,0]) + cos(t[i+1,0]))
                 for i in range(N)])
        
        constraints = []
        for i in range(N):
            constraints.append(x[i+1,0] - x[i,0] - 0.5*h*(sin(t[i+1,0])+sin(t[i,0])) == 0)
            constraints.append(t[i+1,0] - t[i,0] - 0.5*h*(u[i+1,0] - u[i,0]) == 0)
        constraints.append(t <= 1)
        constraints.append(t >= -1)
        constraints.append(-0.05 <= x)
        constraints.append(x <= 0.05)

        p = devkit.model.Model(minimize(f), constraints)

        try:
            info = p.solve(solver=devkit.solver.SolverIpopt(), parameters={'quiet': True})
        except ImportError:
            raise unittest.SkipTest('ipopt not available')
        
        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(f.get_value(), 350.)

    def test_solve_NLP_rosenbrock(self):

        N = 500

        x = devkit.model.VariableMatrix(name='x', shape=(N,1))
        
        f = 0.
        for i in range(N-1):
            f = f + 100*(x[i+1,0]-x[i,0]*x[i,0]) * (x[i+1,0] - x[i,0]*x[i,0]) + (1-x[i,0])*(1-x[i,0])
    
        p = devkit.model.Model(minimize(f))

        try:
            info = p.solve(solver=devkit.solver.SolverIpopt(), parameters={'quiet': True, 'max_iter': 1500})
        except ImportError:
            raise unittest.SkipTest('ipopt not available')

        self.assertEqual(info['status'], 'solved')
        self.assertAlmostEqual(f.get_value(), 0.)
        self.assertTrue(np.all(np.abs(x.get_value()-1.) < 1e-10))
