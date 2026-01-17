import torch
import time
import os
os.system("pip install . --no-build-isolation")
import math
import os
import sys
import numpy as np
import scipy.sparse

# --- Auto-Install Dependencies ---
def install_plotting_deps():
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Installing plotting dependencies...")
        os.system(f"{sys.executable} -m pip install pandas matplotlib seaborn scipy --no-build-isolation")
        print("Dependencies installed.")

install_plotting_deps()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import your extension
import bcsr 

# Ensure reproducibility
torch.manual_seed(42)
BLOCK_SIZE = 4  # The block size your kernels are optimized for

# --- 1. COO -> BCSR Logic (The Python Preprocessor) ---

def coo_to_metal_bcsr(indices, values, num_rows, num_cols, block_size=4):
    """
    Converts PyTorch COO tensors into Block CSR format using Scipy.
    Handles padding logic automatically.
    """
    # 1. Convert to Scipy BSR (Block Sparse Row)
    # This handles the complex logic of densifying blocks and padding
    coo = scipy.sparse.coo_matrix(
        (values.numpy(), (indices[0].numpy(), indices[1].numpy())), 
        shape=(num_rows, num_cols)
    )
    
    # Scipy does the heavy lifting here
    bsr = coo.tobsr(blocksize=(block_size, block_size))
    
    # 2. Extract Flattened Arrays for C++
    # bsr.data is (n_blocks, R, C). We flatten to (n_blocks * R * C)
    flat_vals = torch.from_numpy(bsr.data.flatten()).float()
    
    # bsr.indptr -> Block Row Pointers
    row_ptr = torch.from_numpy(bsr.indptr).int()
    
    # bsr.indices -> Block Column Indices
    col_ind = torch.from_numpy(bsr.indices).int()
    
    # 3. Calculate Padded Dimensions
    # The solver MUST use these for vector allocations
    padded_rows = bsr.shape[0]
    padded_cols = bsr.shape[1]
    
    return flat_vals, row_ptr, col_ind, padded_rows, padded_cols

# --- 2. The Solver Wrapper (Handles Padding Safety) ---

class MetalBCSRSolver:
    def __init__(self, indices, values, rows, cols, block_size=4):
        self.rows = rows
        self.cols = cols
        
        # 1. Convert to BCSR
        t0 = time.time()
        b_vals, b_rptr, b_cind, p_rows, p_cols = coo_to_metal_bcsr(
            indices, values, rows, cols, block_size
        )
        self.convert_time = time.time() - t0
        
        self.padded_rows = p_rows
        self.padded_cols = p_cols
        
        # 2. Initialize C++ Extension
        dummy = torch.empty(0)
        self.cpp_solver = bcsr.bcsr_tensor(
            dummy,      # keys (unused)
            b_vals,     # dense blocks
            b_rptr,     # row_ptr
            b_cind,     # col_ind
            rows,       # Logical rows
            cols,       # Logical cols
            block_size
        )

    def solve(self, b_vector):
        """
        Solves Ax = b, handling input/output padding safety.
        """
        # --- INPUT SAFETY: PAD 'b' ---
        # If logical size is 10 but padded is 12, we must pass 12 to GPU.
        if b_vector.size(0) < self.padded_rows:
            b_padded = torch.zeros(self.padded_rows, dtype=torch.float32)
            b_padded[:self.rows] = b_vector
        else:
            b_padded = b_vector

        # --- OUTPUT PREP ---
        # The C++ code copies back 'logical_cols', so x_out can be logical size.
        x_out = torch.zeros(self.cols, dtype=torch.float32)

        # --- RUN SOLVER ---
        self.cpp_solver.bicgstab(b_padded, x_out)
        
        return x_out

# --- 3. PyTorch Baseline ---

@torch.jit.script
def bicgstab_torch(A, b, max_iter: int = 1000, tol: float = 1e-5):
    x = torch.zeros_like(b)
    r = b - torch.mv(A, x)
    r0_hat = r.clone()
    
    rho_old = 1.0
    alpha = 1.0
    omega = 1.0
    v = torch.zeros_like(b)
    p = torch.zeros_like(b)
    
    for i in range(max_iter):
        rho = torch.dot(r0_hat, r)
        beta = (rho / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)
        
        v = torch.mv(A, p)
        sigma = torch.dot(r0_hat, v)
        
        # Stability check
        if sigma.abs() < 1e-15:
            break
            
        alpha = rho / sigma
        s = r - alpha * v
        
        if torch.norm(s) < tol:
            x = x + alpha * p
            break
            
        t = torch.mv(A, s)
        omega = torch.dot(t, s) / torch.dot(t, t)
        
        x = x + alpha * p + omega * s
        r = s - omega * t
        
        if torch.norm(r) < tol:
            break
            
        rho_old = rho
        
    return x

# --- 4. Matrix Generators ---

def generate_ml_block_sparse(rows, cols, block_size=32, density=0.05):
    """Generates a Block-Sparse Diagonally Dominant Matrix."""
    n_blocks_row = (rows + block_size - 1) // block_size
    n_blocks_col = (cols + block_size - 1) // block_size
    
    # Block Mask
    block_mask = torch.rand(n_blocks_row, n_blocks_col) < density
    
    # Upscale mask
    mask = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    mask = mask[:rows, :cols]
    
    # Random Values + Diagonal Dominance
    A_dense = torch.randn(rows, cols) * mask.float()
    row_sums = torch.sum(torch.abs(A_dense), dim=1)
    
    # Add to diagonal safely
    ids = torch.arange(rows)
    A_dense[ids, ids] += row_sums + 1.0
    
    return A_dense.to_sparse_coo().coalesce(), A_dense

def generate_physics_stencil(grid_n):
    """Generates a 5-point 2D Laplacian Stencil."""
    N = grid_n * grid_n
    rows = torch.arange(N)
    
    # Indices
    indices = []
    values = []
    
    # Center
    indices.append(torch.stack([rows, rows]))
    values.append(torch.full((N,), 4.0))
    
    # Right Neighbor
    mask_r = (rows % grid_n) != (grid_n - 1)
    r_r = rows[mask_r]
    c_r = r_r + 1
    indices.append(torch.stack([r_r, c_r])); values.append(torch.full((r_r.shape[0],), -1.0))
    indices.append(torch.stack([c_r, r_r])); values.append(torch.full((r_r.shape[0],), -1.0))

    # Down Neighbor
    mask_d = rows < (N - grid_n)
    r_d = rows[mask_d]
    c_d = r_d + grid_n
    indices.append(torch.stack([r_d, c_d])); values.append(torch.full((r_d.shape[0],), -1.0))
    indices.append(torch.stack([c_d, r_d])); values.append(torch.full((r_d.shape[0],), -1.0))
    
    all_ind = torch.cat(indices, dim=1)
    all_val = torch.cat(values)
    
    A_sp = torch.sparse_coo_tensor(all_ind, all_val, (N, N)).coalesce()
    return A_sp

# --- 5. Main Benchmark Logic ---

def run_benchmark():
    results = []
    print(f"{'Mode':<10} | {'Size':<10} | {'Custom(s)':<10} | {'PyTorch(s)':<10} | {'Speedup':<8} | {'Error'}")
    print("-" * 75)

    tasks = [
        ("physics", 64),   # 4096 rows
        ("physics", 100),  # 10000 rows
        ("physics", 128),  # 16384 rows
        ("ml", 2048),
        ("ml", 4096),
        ("ml", 8192)
    ]

    for mode, size in tasks:
        try:
            # 1. Generate Data
            if mode == "ml":
                A_sparse, A_dense = generate_ml_block_sparse(size, size, block_size=BLOCK_SIZE, density=0.02)
            else:
                grid_n = size
                size = grid_n * grid_n
                A_sparse = generate_physics_stencil(grid_n)
                A_dense = A_sparse.to_dense()

            b = torch.randn(size)
            
            # 2. PyTorch Baseline (JIT)
            t0 = time.time()
            x_torch = bicgstab_torch(A_dense, b)
            t_torch = time.time() - t0
            
            # 3. Custom Metal BCSR
            # Initialize (includes Conversion overhead)
            solver_wrapper = MetalBCSRSolver(
                A_sparse.indices(), 
                A_sparse.values(), 
                size, size, 
                block_size=BLOCK_SIZE
            )
            
            # Solve Time (Average of 5)
            # Warmup
            _ = solver_wrapper.solve(b)
            
            t_start = time.time()
            for _ in range(5):
                x_custom = solver_wrapper.solve(b)
            t_custom = (time.time() - t_start) / 5.0
            
            # 4. Check Accuracy
            res_custom = torch.norm(b - torch.matmul(A_dense, x_custom)) / torch.norm(b)
            
            speedup = t_torch / t_custom
            print(f"{mode:<10} | {size:<10} | {t_custom:<10.5f} | {t_torch:<10.5f} | {speedup:<8.2f}x | {res_custom:.2e}")
            
            results.append({
                "Mode": mode,
                "Size": size,
                "Custom Time": t_custom,
                "PyTorch Time": t_torch,
                "Speedup": speedup
            })

        except Exception as e:
            print(f"Failed on {mode} {size}: {e}")

    # --- PLOTTING ---
    if not results: return

    df = pd.DataFrame(results)
    
    # Plot Speedup
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Size", y="Speedup", hue="Mode", palette="viridis")
    plt.title(f"Speedup: Metal BCSR (Block={BLOCK_SIZE}) vs PyTorch JIT")
    plt.axhline(1.0, color='red', linestyle='--')
    plt.yscale("log")
    plt.ylabel("Speedup (Log Scale)")
    plt.savefig("benchmark_bcsr_speedup.png")
    
    # Plot Execution Time
    plt.figure(figsize=(10, 6))
    df_melt = df.melt(id_vars=["Size", "Mode"], value_vars=["Custom Time", "PyTorch Time"], var_name="Impl", value_name="Time")
    sns.lineplot(data=df_melt, x="Size", y="Time", hue="Impl", style="Mode", markers=True)
    plt.yscale("log")
    plt.title("Execution Time Comparison")
    plt.ylabel("Time (s)")
    plt.savefig("benchmark_bcsr_time.png")
    
    print("\n[Done] Results saved to benchmark_speedup.png / benchmark_time.png")

if __name__ == "__main__":
    os.system("pip install . --no-build-isolation")
    run_benchmark()