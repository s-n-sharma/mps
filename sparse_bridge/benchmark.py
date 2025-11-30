import torch
import time
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from torch.utils.cpp_extension import load

print("Compiling Extension...")
mps_sparse = load(
    name="mps_sparse_class",
    sources=["bridge.cpp"],
    extra_ldflags=["-framework", "Accelerate"],
    verbose=False
)

def generate_random_data(n, density=0.01):
    """Generates a random sparse matrix A"""
    # Generate random sparse matrix (CSC format)
    A_scipy = scipy.sparse.random(n, n, density=density, format='csc', dtype=np.float64)
    A_scipy = A_scipy + scipy.sparse.eye(n, dtype=np.float64)
    return A_scipy

def generate_data(n):
    """Generates a Laplacian Matrix (Tridiagonal)"""
    # Main diagonal (value 2)
    data = np.ones(n) * 2
    rows = np.arange(n)
    cols = np.arange(n)
    
    # Off diagonals (value -1)
    # Upper
    data = np.concatenate([data, np.ones(n-1) * -1])
    rows = np.concatenate([rows, np.arange(n-1)])
    cols = np.concatenate([cols, np.arange(1, n)])
    
    # Lower
    data = np.concatenate([data, np.ones(n-1) * -1])
    rows = np.concatenate([rows, np.arange(1, n)])
    cols = np.concatenate([cols, np.arange(n-1)])
    
    # Construct
    A = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc()
    A = A + scipy.sparse.eye(n, dtype=np.float64) * 0.1
    return A

def benchmark():
    # Test Config
    sizes = [1000, 4000, 10000, 20000, 100000]
    batch_size = 128 
    
    print(f"\n{'='*85}")
    print(f"{'SIZE':<8} | {'PHASE':<15} | {'SCIPY (ms)':<12} | {'APPLE (ms)':<12} | {'SPEEDUP':<10}")
    print(f"{'='*85}")

    for n in sizes:
        # --- PREPARATION ---
        A_scipy = generate_data(n)
        
        # Convert to PyTorch tensors
        values = torch.from_numpy(A_scipy.data)
        row_indices = torch.from_numpy(A_scipy.indices).long()
        col_starts = torch.from_numpy(A_scipy.indptr).long()
        
        # Create Dummy Data for Solve
        B_batch_np = np.random.rand(n, batch_size).astype(np.float64)
        B_batch_torch = torch.from_numpy(B_batch_np.T).contiguous().t()

        # ---------------------------------------------------------
        # 1. SCIPY BENCHMARK (Using SuperLU with Caching)
        # ---------------------------------------------------------
        
        # A. Factorization
        start = time.perf_counter()
        scipy_solver = scipy.sparse.linalg.splu(A_scipy) 
        scipy_fact_time = (time.perf_counter() - start) * 1000

        # B. Batch Solve
        start = time.perf_counter()
        _ = scipy_solver.solve(B_batch_np)
        scipy_solve_time = (time.perf_counter() - start) * 1000

        # ---------------------------------------------------------
        # 2. APPLE BENCHMARK (Your C++ Class)
        # ---------------------------------------------------------
        
        # A. Factorization
        start = time.perf_counter()
        apple_solver = mps_sparse.SparseQRSolver(col_starts, row_indices, values, n, n)
        apple_fact_time = (time.perf_counter() - start) * 1000

        # B. Batch Solve
        if n == sizes[0]:
            apple_solver.solve(B_batch_torch)

        start = time.perf_counter()
        x_apple = apple_solver.solve(B_batch_torch)
        apple_solve_time = (time.perf_counter() - start) * 1000

        # ---------------------------------------------------------
        # 3. REPORTING
        # ---------------------------------------------------------
        fact_speedup = scipy_fact_time / apple_fact_time
        solve_speedup = scipy_solve_time / apple_solve_time
        
        # Verify Accuracy
        diff = np.linalg.norm(scipy_solver.solve(B_batch_np) - x_apple.numpy())
        status = "PASS" if diff < 1e-8 else f"FAIL ({diff:.1e})"

        print(f"{n:<8} | {'Factorization':<15} | {scipy_fact_time:<12.2f} | {apple_fact_time:<12.2f} | {fact_speedup:<9.2f}x")
        print(f"{'':<8} | {'Batch Solve':<15} | {scipy_solve_time:<12.2f} | {apple_solve_time:<12.2f} | {solve_speedup:<9.2f}x  {status}")
        print("-" * 85)

if __name__ == "__main__":
    benchmark()