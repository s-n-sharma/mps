#include <torch/extension.h>
#include <Accelerate/Accelerate.h>
#include <vector>
#include <iostream>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DOUBLE(x) TORCH_CHECK(x.dtype() == torch::kFloat64, #x " must be float64")
#define CHECK_LONG(x) TORCH_CHECK(x.dtype() == torch::kInt64, #x " must be int64")

void print_accelerate_error(const char * _Nonnull msg) {
    std::cerr << "accelerate" << msg << std::endl;
}

class SparseQRSolver {
    private:
    SparseOpaqueSymbolicFactorization symbolic;
    SparseOpaqueFactorization_Double numeric;

    int rows;
    int cols;

    public:
    SparseQRSolver(
        torch::Tensor col_starts,   
        torch::Tensor row_indices,  
        torch::Tensor values,       
        int64_t n_rows,
        int64_t n_cols): rows((int) n_rows), cols((int) n_cols) {
        
        //safety checks
        CHECK_CONTIGUOUS(col_starts); CHECK_LONG(col_starts);
        CHECK_CONTIGUOUS(row_indices); CHECK_LONG(row_indices);
        CHECK_CONTIGUOUS(values); CHECK_DOUBLE(values);

        //index casting
        long nnz = values.numel();
        std::vector<int> row_indices_int32(nnz);
        int64_t* row_ptr_64 = row_indices.data_ptr<int64_t>();
        for (int i = 0; i < nnz; i++) {
            row_indices_int32[i] = static_cast<int>(row_ptr_64[i]);
        }

        //create structure

        SparseMatrixStructure structure = {
            .rowCount = (int)rows,
            .columnCount = (int)cols,
            .columnStarts = (long*)col_starts.data_ptr<int64_t>(),
            .rowIndices = row_indices_int32.data(),
            .attributes = { .kind = SparseOrdinary },
            .blockSize = 1
        };
        SparseMatrix_Double A = { .structure = structure, .data = values.data_ptr<double>() };

        //symbolic factorization
        SparseSymbolicFactorOptions sym_opts = {
        .control = SparseDefaultControl,
        .orderMethod = SparseOrderDefault,
        .order = NULL,
        .ignoreRowsAndColumns = NULL,
        .malloc = malloc, 
        .free = free,
        .reportError = print_accelerate_error,
        };
        
        this->symbolic = SparseFactor(SparseFactorizationQR, A.structure, sym_opts);

        //TORCH_CHECK(this->symbolic.status, "symbolic failed");

        //numeric factorization

        SparseNumericFactorOptions num_opts = {
            .control = SparseDefaultControl,
            .pivotTolerance = 0.1, 
            .zeroTolerance = 0.0,
        };
        this->numeric = SparseFactor(this->symbolic, A, num_opts);
        //TORCH_CHECK(this->numeric.status, "numeric failed");

        this->rows = (int) n_rows;
        this->cols = (int) n_cols;
}

    ~SparseQRSolver() {
        SparseCleanup(this->numeric);
        SparseCleanup(this->symbolic);
    }
    
    torch::Tensor solve(torch::Tensor B) {
        CHECK_DOUBLE(B); 
        TORCH_CHECK(B.size(0) == this->rows, "B row size mismatch");

        TORCH_CHECK(B.device().is_cpu(), "Input B must be on CPU.");

        // Determine if Vector or Matrix
        int b_rows = B.size(0);


        int b_cols = 1; 
        if (B.dim() > 1) {
            b_cols = B.size(1);
        }

        TORCH_CHECK(b_rows == rows, "bad dim b: " + std::to_string(b_rows) + " , a: " + std::to_string(rows));

       
        auto X = torch::zeros_like(B);

        if (b_cols == 1) {
        //single right hand side
        CHECK_CONTIGUOUS(B);
        DenseVector_Double b_vec = { .data = B.data_ptr<double>(), .count = (int)this->rows, };
        DenseVector_Double x_vec = { .data = X.data_ptr<double>(), .count = (int)this->cols, };

        SparseSolve(this->numeric, b_vec, x_vec);

        } else {

            //batch matrix
            TORCH_CHECK(B.stride(0) == 1, 
                "column major b");

            long col_stride = B.stride(1);

            DenseMatrix_Double b_mat = { 
                .rowCount = (int)this->rows,
                .columnCount = (int)b_cols,
                .columnStride = (int)col_stride, 
                .attributes = {0},
                .data = B.data_ptr<double>()
            }; 

            if (X.stride(0) != 1) {
                X = X.t().contiguous().t(); 
            }
            long x_col_stride = X.stride(1);

            DenseMatrix_Double x_mat = { 
                .rowCount = (int)this->cols,
                .columnCount = (int)b_cols,
                .columnStride = (int)x_col_stride,
                .attributes = {0},
                .data = X.data_ptr<double>()
            };

            SparseSolve(this->numeric, b_mat, x_mat);
        }

        return X;
    }

};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<SparseQRSolver>(m, "SparseQRSolver")
        .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t>())
        .def("solve", &SparseQRSolver::solve);
}

