from pyProblem import Problem
import pyVector as pyVec
import pyOperator as pyOp
from math import isnan

#TODO: if you desire to create a new problem class for the weighted optimization,
#      you will need to modify this file (Hint: if you want to do the least amount of work, only
#      one function needs to be modified although there are other ways to do it.)

class ProblemL2LinearRegWgt(Problem):
    """Weighted linear inverse problem regularized of the form 1/2*|W(Lm-d)|_2 + epsilon^2/2*|Am-m_prior|_2"""
    
    def __init__(self, model, data, op, epsilon, wgt_op=None, reg_op=None, prior_model=None, prec=None,
                 minBound=None, maxBound=None, boundProj=None):
        """
        Constructor of linear regularized problem:
        model    	= [no default] - vector class; Initial model vector
        data     	= [no default] - vector class; Data vector
        op       	= [no default] - linear operator class; L operator
        epsilon      = [no default] - float; regularization weight
        wgt_op       = [Identity] - linear operator class; A weighting operator
        reg_op       = [Identity] - linear operator class; A regularization operator
        prior_model  = [None] - vector class; Prior model for regularization term
        minBound		= [None] - vector class; Minimum value bounds
        maxBound		= [None] - vector class; Maximum value bounds
        boundProj	= [None] - Bounds class; Class with a function "apply(input_vec)" to project input_vec onto some convex set
        prec       	= [None] - linear operator class; Preconditioning matrix
        """
        # Setting the bounds (if any)
        super(ProblemL2LinearRegWgt, self).__init__(minBound, maxBound, boundProj)
        self.op=op
        # Setting internal vector
        self.model = model.clone()
        self.dmodel = model.clone()
        self.dmodel.zero()
        # Gradient vector
        self.grad = self.dmodel.clone()
        # Copying the pointer to data vector
        self.data = data
        # Setting a prior model (if any)
        self.prior_model = prior_model
        # Setting linear operators
        # Assuming identity operator if regularization operator was not provided
        if reg_op is None:
            reg_op = pyOp.IdentityOp(self.model)
        # Checking if space of the prior model is constistent with range of
        # regularization operator
        if self.prior_model is not None:
            if not self.prior_model.checkSame(reg_op.range):
                raise ValueError("Prior model space no constistent with range of regularization operator")
        self.epsilon = epsilon  # Regularization 
        # Residual vector (data and model residual vectors)
        self.res = self.op.range.clone()
        self.res.zero()
        # Dresidual vector
        self.dres = self.res.clone()
        # Setting default variables
        self.setDefaults()
        self.linear = True
        # Preconditioning matrix
        self.prec = prec
        # Objective function terms (useful to analyze each term)
        self.obj_terms = [None, None]
    
    def __del__(self):
        """Default destructor"""
        return
    
    def resf(self, model):
        """Method to return residual vector r = [r_d; r_m]: r_d = W(Lm - d); r_m = epsilon * (Am - m_prior) """
        if model.norm() != 0.:
            self.op.forward(False, model, self.res)
        else:
            self.res.zero()
        # Computing r_d = Lm - d
        self.res.vecs[0].scaleAdd(self.wdata, 1., -1.)
        # Computing r_m = Am - m_prior
        if self.prior_model is not None:
            self.res.vecs[1].scaleAdd(self.prior_model, 1., -1.)
        # Scaling by epsilon epsilon*r_m
        self.res.vecs[1].scale(self.epsilon)
        return self.res
    
    def gradf(self, model, res):
        """Method to return gradient vector g = W'L'r_d + epsilon*A'r_m"""
        # Scaling by epsilon the model residual vector (saving temporarily residual regularization)
        # g = epsilon*A'r_m
        self.op.ops[1].adjoint(False, self.grad, res.vecs[1])
        self.grad.scale(self.epsilon)
        # g = L'r_d + epsilon*A'r_m
        self.op.ops[0].adjoint(True, self.grad, res.vecs[0])
        return self.grad
    
    def dresf(self, model, dmodel):
        """Method to return residual vector dres = (WL + epsilon * A)dm"""
        # Computing Ldm = dres_d
        self.op.forward(False, dmodel, self.dres)
        # Scaling by epsilon
        self.dres.vecs[1].scale(self.epsilon)
        return self.dres
    
    def objf(self, res):
        """Method to return objective function value 1/2|W(Lm-d)|_2 + epsilon^2/2*|Am-m_prior|_2"""
        for idx in range(res.n):
            val = res.vecs[idx].norm()
            self.obj_terms[idx] = 0.5 * val*val
        return sum(self.obj_terms)

