import numpy as np
import lightgbm as lgb


from .lambdaobj import get_unbiased_gradients_fixed_t, get_unbiased_gradients
from .calculator import Calculator, MIN_ARG, MAX_ARG



class DatasetWithCalculatorRanks0AndT(lgb.Dataset):
    def __init__(
        self, 
        max_ndcg_pos, 
        ranks, 
        eta, 
        inverse_max_dcgs=None, 
        sigmoids=None,
        logs=None,
        idx_factor=None,
        *args, 
        **kwargs
    ):
        lgb.Dataset.__init__(self, *args, **kwargs)
        self.calculator = Calculator(
            self.label, 
            self.get_group(), 
            max_ndcg_pos,
            inverse_max_dcgs, 
            sigmoids,
            logs,
            idx_factor
        )
        self.ranks0 = (ranks - 1).astype(int)
        self.t_plus = 1 / np.power(np.arange(1, self.calculator.max_rank + 1), eta)
        self.t_minus = np.ones(self.calculator.max_rank)


class DatasetWithCalculatorRanks0AndP(lgb.Dataset):
    def __init__(
        self, 
        max_ndcg_pos, 
        ranks,
        p,
        inverse_max_dcgs=None, 
        sigmoids=None,
        logs=None,
        idx_factor=None,
        *args, 
        **kwargs
    ):
        lgb.Dataset.__init__(self, *args, **kwargs)
        self.calculator = Calculator(
            self.label, 
            self.get_group(), 
            max_ndcg_pos,
            inverse_max_dcgs, 
            sigmoids,
            logs,
            idx_factor
        )
        self.ranks0 = (ranks - 1).astype(int)
        self.p = p
        self.t_plus = np.ones(self.calculator.max_rank)
        self.t_minus = np.ones(self.calculator.max_rank)


def unbiased_lambdarank_objective_fixed_t(preds, dataset):
    groups = dataset.get_group()
    
    if len(groups) == 0:
        raise Error("Group/query data should not be empty.")
    else:
        grad = np.zeros(len(preds))
        hess = np.zeros(len(preds))
        get_unbiased_gradients_fixed_t(
            np.ascontiguousarray(dataset.label, dtype=np.double), 
            np.ascontiguousarray(preds),
            np.ascontiguousarray(dataset.t_plus),
            np.ascontiguousarray(dataset.t_minus),
            np.ascontiguousarray(dataset.ranks0),
            len(preds),
            np.ascontiguousarray(groups),
            np.ascontiguousarray(dataset.calculator.query_boundaries),
            len(dataset.calculator.query_boundaries) - 1,
            np.ascontiguousarray(dataset.calculator.discounts),
            np.ascontiguousarray(dataset.calculator.inverse_max_dcgs),
            np.ascontiguousarray(dataset.calculator.sigmoids),
            len(dataset.calculator.sigmoids),
            MIN_ARG,
            MAX_ARG,
            dataset.calculator.idx_factor,
            np.ascontiguousarray(grad), 
            np.ascontiguousarray(hess)
        )
        
        return grad, hess


def unbiased_lambdarank_objective(preds, dataset):
    groups = dataset.get_group()
    
    if len(groups) == 0:
        raise Error("Group/query data should not be empty.")
    else:
        grad = np.zeros(len(preds))
        hess = np.zeros(len(preds))
        get_unbiased_gradients(
            np.ascontiguousarray(dataset.label, dtype=np.double), 
            np.ascontiguousarray(preds),
            np.ascontiguousarray(dataset.ranks0),
            len(preds),
            np.ascontiguousarray(groups),
            np.ascontiguousarray(dataset.calculator.query_boundaries),
            len(dataset.calculator.query_boundaries) - 1,
            np.ascontiguousarray(dataset.calculator.discounts),
            np.ascontiguousarray(dataset.calculator.inverse_max_dcgs),
            np.ascontiguousarray(dataset.calculator.sigmoids),
            len(dataset.calculator.sigmoids),
            MIN_ARG,
            MAX_ARG,
            dataset.calculator.idx_factor,
            np.ascontiguousarray(dataset.calculator.logs),
            len(dataset.calculator.logs),
            MIN_ARG,
            MAX_ARG,
            dataset.calculator.idx_factor,
            dataset.p,
            dataset.calculator.max_rank,
            np.ascontiguousarray(grad), 
            np.ascontiguousarray(hess),
            np.ascontiguousarray(dataset.t_plus),
            np.ascontiguousarray(dataset.t_minus)
        )
        
        return grad, hess