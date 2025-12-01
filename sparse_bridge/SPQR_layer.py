#wrapper for sparse QR solver
import torch
from torch.utils.cpp_extension import load

mps_sparse = load(
    name="mps_sparse", 
    sources=["bridge.cpp"],
    extra_ldflags=["-framework", "Accelerate"],
    verbose=False
) 

class SparseSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, solver, A_sparse, b):

        x = solver.solve(b)
        ctx.save_for_backward(A_sparse, x)
        ctx.solver = solver

        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        A_sparse, x = ctx.saved_tensors
        solver = ctx.solver

        A_t = A_sparse.t().coalesce()
        A_t_csc = A_t.to_sparse_csc()

        #need to create solver for transpose pass
        
        t_solver = mps_sparse.SparseQRSolver(A_t_csc.ccol_indices(), A_t_csc.row_indices(), A_t_csc.values(), A_t.shape[0], A_t.shape[1])
        
        #compute gradient wrt b: dL/db = A^{-T} * dL/dx
        # A^T dL/dB = dL/dx -> Solve A^T y = c system, y is dL/dB


        g_in = grad_output.cpu()
        if g_in.dim() > 1 and g_in.stride(0) != 1:
            g_in = g_in.t().contiguous().t()
        
        grad_b = t_solver.solve(g_in)
            

        #computute gradient wrt A: dL/dA = - A^{-T} * dL/dx * x^T = - dL/db * x^T

        if ctx.needs_input_grad[1]:
            if grad_b.dim() > 1:
                grad_a_sparse = None
                #need to implement batched gradient
            else:
                grad_a = solver.get_grad_a_helper(A_sparse.to_sparse_csc().ccol_indices(), A_sparse.to_sparse_csc().row_indices(), grad_b, x.cpu(), A_sparse.shape[0], A_sparse.shape[1])
                grad_a_sparse = torch.sparse_coo_tensor(A_sparse.indices(), grad_a, A_sparse.size())

        else:
            grad_a_sparse = None

        return None, grad_a_sparse, grad_b

            



        





