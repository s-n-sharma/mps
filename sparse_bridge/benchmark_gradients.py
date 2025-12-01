import torch
import time
import numpy as np
import scipy.sparse
from torch.utils.cpp_extension import load
from SPQR_layer import SparseSolve, mps_sparse



def generate_data(n, density=None):
    """
    Generates a 1D Laplacian.
    """

    main_diag = np.ones(n) * 2.0
    off_diag = np.ones(n-1) * -1.0
    
    A_scipy = scipy.sparse.diags(
        [off_diag, main_diag, off_diag], 
        [-1, 0, 1], 
        shape=(n, n), 
        format='csc'
    )
    
    A_scipy = A_scipy + scipy.sparse.eye(n) * 0.1
    
    values = torch.from_numpy(A_scipy.data)
    rows = torch.from_numpy(A_scipy.indices).long()
    cols = torch.from_numpy(A_scipy.indptr).long()
    
    coo = A_scipy.tocoo()
    A_coo = torch.sparse_coo_tensor(
        torch.stack([torch.from_numpy(coo.row), torch.from_numpy(coo.col)]), 
        torch.from_numpy(coo.data), 
        (n, n)
    ).coalesce()

    return A_coo, cols, rows, values

def benchmark():
    sizes = [1000, 2000, 4000, 10000, 20000]
    
    print(f"\n{'='*80}")
    print(f"{'SIZE':<6} | {'METHOD':<10} | {'FWD (ms)':<10} | {'BWD (ms)':<10} | {'TOTAL (ms)':<10}")
    print(f"{'='*80}")

    for n in sizes:
        # --- Prepare Data ---
        A_sparse, c_ptr, r_ptr, vals = generate_data(n)
        
        # Inputs requiring grad
        A_sparse.requires_grad_(True)
        b = torch.randn(n, dtype=torch.float64, requires_grad=True)
        target = torch.randn(n, dtype=torch.float64)

        # ----------------------------------------
        # 1. PyTorch Dense Baseline
        # ----------------------------------------
        if n <= 8000: 
            A_dense = A_sparse.to_dense().clone().detach().requires_grad_(True)
            b_dense = b.clone().detach().requires_grad_(True)
            
            # Forward
            t0 = time.time()
            x_dense = torch.linalg.solve(A_dense, b_dense)
            fwd_time = (time.time() - t0) * 1000
            
            # Backward
            loss = torch.nn.functional.mse_loss(x_dense, target)
            t0 = time.time()
            loss.backward()
            bwd_time = (time.time() - t0) * 1000
            
            print(f"{n:<6} | {'Dense':<10} | {fwd_time:<10.2f} | {bwd_time:<10.2f} | {fwd_time+bwd_time:<10.2f}")
        else:
             print(f"{n:<6} | {'Dense':<10} | {'OOM':<10} | {'OOM':<10} | {'OOM':<10}")

        # ----------------------------------------
        # 2. Apple Sparse Hybrid
        # ----------------------------------------
        
        # Create Solver (Forward pass setup)
        solver = mps_sparse.SparseQRSolver(c_ptr, r_ptr, vals, n, n)
        
        # Forward
        t0 = time.time()
        # Apply the layer
        x_sparse = SparseSolve.apply(solver, A_sparse, b)
        fwd_time = (time.time() - t0) * 1000
        
        # Backward
        loss = torch.nn.functional.mse_loss(x_sparse, target)
        t0 = time.time()
        loss.backward()
        bwd_time = (time.time() - t0) * 1000
        
        print(f"{n:<6} | {'Sparse':<10} | {fwd_time:<10.2f} | {bwd_time:<10.2f} | {fwd_time+bwd_time:<10.2f}")
        print("-" * 80)

if __name__ == "__main__":
    benchmark()