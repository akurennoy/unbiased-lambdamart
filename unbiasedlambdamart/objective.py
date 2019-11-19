import numpy as np
import lightgbm as lgb


from .lambdaobj import get_unbiased_gradients
from .calculator import Calculator, MIN_ARG, MAX_ARG



class DatasetWithCalculatorRanks0AndT(lgb.Dataset):
    def __init__(
        self, max_ndcg_pos, ranks, *args, **kwargs
    ):
        lgb.Dataset.__init__(self, *args, **kwargs)
        self.calculator = Calculator(self.label, self.get_group(), max_ndcg_pos)
        self.ranks0 = (ranks - 1).astype(int)
        self.t_plus = 1 / np.arange(1, self.calculator.max_rank + 1)
        self.t_minus = np.ones(self.calculator.max_rank)

        
def unbiased_lambdarank_objective(preds, dataset):
    """
    Uses global variables t_plus and t_minus.
    """
    groups = dataset.get_group()
    
    if len(groups) == 0:
        raise Error("Group/query data should not be empty.")
    else:
        grad = np.zeros(len(preds))
        hess = np.zeros(len(preds))
        get_unbiased_gradients(
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