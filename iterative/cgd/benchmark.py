import torch
import spmv  # The name defined in your setup.py
import time
import math
import os
import argparse

def generate_spd_matrix(rows, cols, density=0.01):
    """
    Generates a Sparse Symmetric Positive Definite matrix.
    Method: A = M^T * M + alpha * I
    """
    print(f"Generating random sparse matrix ({rows}x{cols}, density={density})...")
    
    # 1. Generate a random sparse matrix M
    nnz = int(rows * cols * density)
    indices = torch.randint(0, rows, (2, nnz))
    values = torch.rand(nnz)
    M = torch.sparse_coo_tensor(indices, values, (rows, cols))
    
    # 2. Make it dense temporarily to perform A = M^T @ M
    M_dense = M.to_dense()
    A_dense = torch.matmul(M_dense.t(), M_dense)
    
    # 3. Add diagonal regularization to ensure Positive Definiteness
    # Increased to 5.0 to make the matrix better conditioned (more stable)
    A_dense.diagonal().add_(5.0)
    
    # 4. Convert back to sparse COO
    A_sparse = A_dense.to_sparse()
    
    print("Matrix generation complete.")
    return A_sparse, A_dense

def pack_keys(row_indices, col_indices):
    """
    Packs row and column indices into a single uint64 tensor.
    """
    row_indices = row_indices.to(torch.int64)
    col_indices = col_indices.to(torch.int64)
    packed = (row_indices << 32) | col_indices
    return packed

def conjugate_gradient_torch(A, b, x0=None, max_iter=None, tol=1e-3):
    """
    Standard Conjugate Gradient implementation in pure PyTorch.
    Solves Ax = b
    """
    if max_iter is None:
        max_iter = len(b)
        
    x = torch.zeros_like(b) if x0 is None else x0
    
    # r = b - A @ x
    r = b - torch.matmul(A, x)
    p = r.clone()
    rsold = torch.dot(r, r)
    
    for i in range(max_iter):
        # Ap = A @ p
        Ap = torch.matmul(A, p)
        
        alpha = rsold / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        
        rsnew = torch.dot(r, r)
        
        # Check convergence
        if torch.sqrt(rsnew) < tol:
            print(f"   -> PyTorch CG converged in {i+1} iterations.")
            break
            
        p = r + (rsnew / rsold) * p
        rsold = rsnew
        
    return x

def benchmark(n_rows=40108, density=0.05):
    # --- 1. Data Preparation ---
    A_sparse, A_dense = generate_spd_matrix(n_rows, n_rows, density)
    
    indices = A_sparse.indices()
    values = A_sparse.values().to(torch.float32)
    
    row_indices = indices[0]
    col_indices = indices[1]
    nnz = values.shape[0]

    # Create target vector 'b'
    x_true = torch.randn(n_rows, dtype=torch.float32)
    b = torch.matmul(A_dense, x_true)

    # Preparce inputs for C++ Extension
    keys = pack_keys(row_indices, col_indices)
    row_ptr_buffer = torch.zeros(n_rows + 1, dtype=torch.int32)
    col_ind_buffer = torch.zeros(nnz, dtype=torch.int32)
    out_vals_buffer = torch.zeros(nnz, dtype=torch.float32)
    x_sol = torch.zeros(n_rows, dtype=torch.float32)

    print(f"\nInitializing Metal CSR Tensor (N={n_rows}, NNZ={nnz})...")
    
    # --- 2. Initialize Extension ---
    start_init = time.time()
    tensor_impl = spmv.csr_tensor(
        keys,
        values,
        row_ptr_buffer,
        col_ind_buffer,
        out_vals_buffer,
        n_rows,
        n_rows
    )
    end_init = time.time()
    print(f"Initialization (COO->CSR on GPU) took: {(end_init - start_init)*1000:.2f} ms")

    # --- 3. Run Benchmark (Custom Solver) ---
    print("\nStarting Custom Solver Benchmark...")
    
    # Warmup
    x_warmup = torch.zeros_like(x_sol)
    tensor_impl.iter_solve(b, x_warmup)
    
    num_runs = 10
    timings = []
    
    for i in range(num_runs):
        x_sol.zero_()
        t0 = time.time()
        tensor_impl.iter_solve(b, x_sol)
        t1 = time.time()
        timings.append((t1 - t0) * 1000)

    avg_time = sum(timings) / len(timings)
    print(f"Average Solve Time (Custom CGD): {avg_time:.2f} ms")

    # --- 4. Validation ---
    
    # CHECK 1: Did the solver explode?
    if torch.isnan(x_sol).any() or torch.isinf(x_sol).any():
        print("\n❌ CRITICAL: The solver output contains NaNs or Infs.")
    else:
        # CHECK 2: Calculate Residual with Safe Division
        b_pred = torch.matmul(A_dense, x_sol)
        
        epsilon = 1e-8
        numerator = torch.norm(b_pred - b)
        denominator = torch.norm(b) + epsilon
        residual = numerator / denominator
        
        print(f"\n--- Results (Custom) ---")
        print(f"Relative Residual: {residual.item():.6e}")
        
        if residual.item() < 1e-3:
            print("✅ Custom Solver Converged!")
        else:
            print("⚠️ Custom Solver did not converge fully.")

    # --- 5. Comparison (PyTorch CG) ---
    print("\nRunning PyTorch Native Conjugate Gradient (CPU) for comparison...")
    
    # We use A_dense for the PyTorch baseline to ensure standard matmul behavior, 
    # but the algorithm is the same (CG).
    
    t0 = time.time()
    try:
        # Using a relaxed tolerance or fixed max_iter to match standard GPU performance expectations
        # You can adjust max_iter=... to match the iteration count of your C++ solver if known
        x_torch = conjugate_gradient_torch(A_dense, b, max_iter=100, tol=1e-3)
        t1 = time.time()
        
        py_time = (t1-t0)*1000
        print(f"PyTorch Native CG Time: {py_time:.2f} ms")
        print(f"Speedup vs PyTorch: {py_time / avg_time:.2f}x")
        
    except Exception as e:
        print(f"PyTorch CG failed: {e}")

if __name__ == "__main__":
    os.system("pip install . --no-build-isolation") 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=15000)
    parser.add_argument("--density", type=float, default=0.05)
    args = parser.parse_args()
    
    benchmark(args.rows, args.density)