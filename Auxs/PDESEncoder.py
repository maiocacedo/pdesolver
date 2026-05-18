import json
import numpy as np
import sympy as sp

class PDESEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, sp.Basic):
            return str(obj)  
        if hasattr(obj, '__dict__'):
            return obj.__dict__  
        return super().default(obj)
    