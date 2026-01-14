import torch
import spmv  # Your custom extension
import time
import math
import os
import argparse
import sys

# --- Auto-Install Dependencies for Plotting ---
def install_plotting_deps():
    print("Checking and installing plotting dependencies...")
    os.system("pip install pandas matplotlib seaborn --no-build-isolation")

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    install_plotting_deps()
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

# --- Matrix Generators ---

def generate_ml_block_sparse_spd(rows, cols, block_size=32, density=0.05):
    """
    Generates a Block-Sparse SPD Matrix.
    SPD Property: $A = M^T M + \\alpha I$
    """
    # 1. Create a Block Mask
    n_blocks_row = rows // block_size
    n_blocks_col = cols // block_size
    
    block_mask = torch.rand(n_blocks_row, n_blocks_col) < density
    
    # Upscale the mask
    mask_rows = block_mask.repeat_interleave(block_size, dim=0)
    mask_full = mask_rows.repeat_interleave(block_size, dim=1)
    mask_full = mask_full[:rows, :cols]
    
    # 2. Generate random values and apply mask
    M = torch.randn(rows, cols) * mask_full.float()
    
    # 3. Make SPD: A = M^T @ M
    # We do this calculation on CPU carefully to avoid OOM on smaller GPUs if size is large
    A_dense = torch.matmul(M.t(), M)
    
    # 4. Regularize diagonal
    A_dense.diagonal().add_(1.0)
    
    # 5. Sparsify
    A_sparse = A_dense.to_sparse()
    real_density = A_sparse._nnz() / (rows*cols)
    
    return A_sparse, A_dense, real_density

def generate_physics_stencil_spd(grid_size):
    """
    Generates a 5-point Laplacian Stencil.
    Matrix Size: $N = grid\_size^2$
    """
    N = grid_size * grid_size
    
    # Main diagonal (4.0)
    diagonals = [torch.full((N,), 4.0)]
    offsets = [0]
    
    # Off-diagonals (-1.0)
    off_1 = torch.full((N-1,), -1.0)
    for i in range(1, grid_size):
        off_1[i*grid_size - 1] = 0
    diagonals.extend([off_1, off_1])
    offsets.extend([1, -1])
    
    off_grid = torch.full((N - grid_size,), -1.0)
    diagonals.extend([off_grid, off_grid])
    offsets.extend([grid_size, -grid_size])
    
    indices_list = []
    values_list = []
    
    for diag, offset in zip(diagonals, offsets):
        if offset >= 0:
            rows = torch.arange(0, N - offset)
            cols = torch.arange(offset, N)
        else:
            rows = torch.arange(-offset, N)
            cols = torch.arange(0, N + offset)
        indices_list.append(torch.stack([rows, cols]))
        values_list.append(diag)
        
    all_indices = torch.cat(indices_list, dim=1)
    all_values = torch.cat(values_list)
    
    A_sparse = torch.sparse_coo_tensor(all_indices, all_values, (N, N))
    A_dense = A_sparse.to_dense()
    
    return A_sparse, A_dense

# --- Utilities ---

def pack_keys(row_indices, col_indices):
    row_indices = row_indices.to(torch.int64)
    col_indices = col_indices.to(torch.int64)
    return (row_indices << 32) | col_indices

def conjugate_gradient_torch(A, b, max_iter=100, tol=1e-4):
    x = torch.zeros_like(b)
    r = b - torch.matmul(A, x)
    p = r.clone()
    rsold = torch.dot(r, r)
    
    for i in range(max_iter):
        Ap = torch.matmul(A, p)
        alpha = rsold / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r, r)
        if torch.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

# --- Benchmarking Logic ---

def benchmark_single_run(mode, size, density):
    print(f"--- Running: {mode} | Size: {size} | Target Density: {density} ---")
    
    # 1. Generate Matrix
    if mode == "ml":
        A_sparse, A_dense, real_density = generate_ml_block_sparse_spd(size, size, density=density)
    elif mode == "physics":
        grid_size = int(math.sqrt(size))
        A_sparse, A_dense = generate_physics_stencil_spd(grid_size)
        real_density = A_sparse._nnz() / (A_dense.shape[0] * A_dense.shape[1])
        # Update size to actual generated size
        size = A_dense.shape[0]

    rows, cols = A_dense.shape
    indices = A_sparse.coalesce().indices()
    values = A_sparse.coalesce().values().to(torch.float32)
    nnz = values.shape[0]

    # 2. Prepare Data
    x_true = torch.randn(rows, dtype=torch.float32)
    x_true = x_true / torch.norm(x_true)
    b = torch.matmul(A_dense, x_true)
    
    # 3. Custom Extension Init
    keys = pack_keys(indices[0], indices[1])
    row_ptr = torch.zeros(rows + 1, dtype=torch.int32)
    col_ind = torch.zeros(nnz, dtype=torch.int32)
    val_buf = torch.zeros(nnz, dtype=torch.float32)
    x_sol = torch.zeros(rows, dtype=torch.float32)
    
    t_init_start = time.time()
    tensor_impl = spmv.csr_tensor(keys, values, row_ptr, col_ind, val_buf, rows, cols)
    t_init = (time.time() - t_init_start) * 1000

    # 4. Benchmark Custom Solver
    # Warmup
    tensor_impl.iter_solve(b, x_sol)
    
    custom_timings = []
    for _ in range(5):
        x_sol.zero_()
        t0 = time.time()
        tensor_impl.iter_solve(b, x_sol)
        custom_timings.append(time.time() - t0)
    avg_custom = sum(custom_timings) / len(custom_timings)

    # 5. Benchmark PyTorch
    # We restrict PyTorch max_iter to match custom extension logic (usually fixed iter)
    # or give it a fair run.
    t0 = time.time()
    conjugate_gradient_torch(A_dense, b, max_iter=200)
    py_time = time.time() - t0
    
    # 6. Calc Metrics
    speedup = py_time / avg_custom
    
    return {
        "Mode": mode,
        "Size": size,
        "Target Density": density if density else "N/A",
        "Real Density": real_density,
        "NNZ": nnz,
        "Custom Time (ms)": avg_custom * 1000,
        "PyTorch Time (ms)": py_time * 1000,
        "Speedup": speedup
    }

def plot_results(df):
    sns.set_theme(style="whitegrid")
    
    # 1. ML Mode: Time vs Size (Hue: Density)
    ml_data = df[df["Mode"] == "ml"]
    if not ml_data.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot A: Absolute Time
        sns.lineplot(data=ml_data, x="Size", y="Custom Time (ms)", hue="Target Density", 
                     marker="o", palette="viridis", ax=axes[0], label="Custom")
        # We also plot PyTorch time, but it might clutter. Let's do a comparison plot instead.
        # Actually, let's plot Speedup vs Size.
        
        # Plot A: Speedup vs Size
        sns.lineplot(data=ml_data, x="Size", y="Speedup", hue="Target Density", 
                     marker="o", palette="rocket", linewidth=2.5, ax=axes[0])
        axes[0].set_title("ML Matrices: Speedup (Custom vs PyTorch Dense)")
        axes[0].set_ylabel("Speedup Factor (x)")
        axes[0].axhline(1.0, color='gray', linestyle='--')
        
        # Plot B: Absolute Execution Time Comparison (Log Scale)
        # Melt for seaborn
        ml_melt = ml_data.melt(id_vars=["Size", "Target Density"], 
                               value_vars=["Custom Time (ms)", "PyTorch Time (ms)"],
                               var_name="Method", value_name="Time (ms)")
        
        sns.lineplot(data=ml_melt, x="Size", y="Time (ms)", hue="Method", style="Target Density",
                     markers=True, dashes=False, ax=axes[1])
        axes[1].set_yscale("log")
        axes[1].set_title("ML Matrices: Execution Time (Log Scale)")
        
        plt.tight_layout()
        plt.savefig("benchmark_ml_results.png")
        print("Saved benchmark_ml_results.png")

    # 2. Physics Mode
    phy_data = df[df["Mode"] == "physics"]
    if not phy_data.empty:
        plt.figure(figsize=(8, 6))
        
        # Comparison of times
        phy_melt = phy_data.melt(id_vars=["Size"], 
                                 value_vars=["Custom Time (ms)", "PyTorch Time (ms)"],
                                 var_name="Method", value_name="Time (ms)")
        
        sns.barplot(data=phy_melt, x="Size", y="Time (ms)", hue="Method", palette="muted")
        plt.title("Physics (Laplacian) Matrices: Execution Time")
        plt.ylabel("Time (ms)")
        plt.yscale("log")
        
        # Add speedup labels on top
        for i, row in phy_data.iterrows():
            # Find the x-coordinate (bit tricky with bars, so we just print it)
            pass 

        plt.tight_layout()
        plt.savefig("benchmark_physics_results.png")
        print("Saved benchmark_physics_results.png")

def main():
    # Install custom extension
    os.system("pip install . --no-build-isolation")
    
    # Configuration
    sizes = [1024, 5120, 7168, 10240, 15360, 20480]
    densities = [0.05, 0.025, 0.01]
    results = []

    print("\n========================================")
    print("Starting Comprehensive Benchmark Suite")
    print("========================================\n")

    # --- ML Loop ---
    for size in sizes:
        for density in densities:
            try:
                res = benchmark_single_run("ml", size, density)
                results.append(res)
            except Exception as e:
                print(f"Error running ML {size} / {density}: {e}")

    # --- Physics Loop ---
    # Physics density is fixed by the stencil, so we only run once per size
    for size in sizes:
        try:
            res = benchmark_single_run("physics", size, None)
            results.append(res)
        except Exception as e:
            print(f"Error running Physics {size}: {e}")

    # --- Reporting ---
    df = pd.DataFrame(results)
    print("\nBenchmark Complete. Summary:")
    print(df[["Mode", "Size", "Target Density", "Speedup"]])
    
    # Save raw data
    df.to_csv("benchmark_results.csv", index=False)
    
    # Generate Plots
    plot_results(df)

if __name__ == "__main__":
    main()