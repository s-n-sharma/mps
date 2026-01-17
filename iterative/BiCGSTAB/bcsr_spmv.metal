#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

#define BLOCK_SIZE 4
constant int THREADGROUP_SIZE = 256;
constant int SIMD_WIDTH = 32;

kernel void inner_product(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device atomic_float* ret [[buffer(2)]],
    constant uint& size [[buffer(3)]], 
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_simdgroup]]
) {
    //need to optimize via reduction
    if (gid < size) {
        float prod = a[gid] * b[gid];
        float local_sum = simd_sum(prod);

        if (tid == 0) {
            atomic_fetch_add_explicit(ret, local_sum, memory_order_relaxed);
        }
    }
}



kernel void spmv_op(
    device const int* row_ptr [[buffer(0)]],
    device const int* col_ind [[buffer(1)]],
    device const float4* val_blocks [[buffer(2)]], 
    device const float* x [[buffer(3)]],
    device float* b [[buffer(4)]],
    constant uint& num_blk_rows [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_blk_rows) {
        return;
    }

    float4 sum = 0.0; 

    int start = row_ptr[gid];
    int end = row_ptr[gid + 1];

    for (int i = start; i < end; i++) {

        int blk_col = col_ind[i]; 
        
        float4 vec_x = *(device const float4*)(x + blk_col * 4);

        int base_idx = i * 4;
        
        float4 r0 = val_blocks[base_idx];
        float4 r1 = val_blocks[base_idx + 1];
        float4 r2 = val_blocks[base_idx + 2];
        float4 r3 = val_blocks[base_idx + 3];

        sum.x = fma(dot(r0, vec_x), 1.0f, sum.x);
        sum.y = fma(dot(r1, vec_x), 1.0f, sum.y);
        sum.z = fma(dot(r2, vec_x), 1.0f, sum.z);
        sum.w = fma(dot(r3, vec_x), 1.0f, sum.w);

    }

    *(device float4*)(b + gid * 4) = sum;
}

kernel void fused_spvmv(
    device const int* row_ptr [[buffer(0)]],
    device const int* col_ind [[buffer(1)]],
    device const float* val_blocks [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device const float* y [[buffer(4)]],
    device float* b [[buffer(5)]], 
    device atomic_float* ret [[buffer(6)]], 
    constant uint& num_blk_rows [[buffer(7)]], 
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_simdgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_position_in_threadgroup]]
) {
    float4 sum = 0.0; 
    
    if (gid < num_blk_rows) {
        int start = row_ptr[gid];
        int end = row_ptr[gid + 1];

        for (int i = start; i < end; i++) {
            int blk_col = col_ind[i];
            
            float4 r0 = *(device const float4*)(val_blocks + 16*i);
            float4 r1 = *(device const float4*)(val_blocks + 16*i + 4);
            float4 r2 = *(device const float4*)(val_blocks + 16*i + 8);
            float4 r3 = *(device const float4*)(val_blocks + 16*i + 12);

            float4 vec_x = *(device const float4*)(x + 4 * blk_col);

            sum.x = sum.x + dot(r0, vec_x);
            sum.y = sum.y + dot(r1, vec_x);
            sum.z = sum.z + dot(r2, vec_x);
            sum.w = sum.w + dot(r3, vec_x);
        }
        
        *(device float4*)(b + 4 * gid) = sum;
    }

    float par_dot = 0.0;
    if (gid < num_blk_rows) {
        float4 vec_y = *(device const float4*)(y + 4 * gid);
        par_dot = dot(sum, vec_y);
    }


    float par_sum = simd_sum(par_dot);

    threadgroup float temp[8]; 

    if (tid == 0) {
        temp[sid] = par_sum;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        float thread_group_sum = 0.0;

        for (int i = 0; i < THREADGROUP_SIZE/SIMD_WIDTH; i++) {
            thread_group_sum += temp[i];
        }

        if (thread_group_sum != 0.0 && !isnan(thread_group_sum)) {
            atomic_fetch_add_explicit(ret, thread_group_sum, memory_order_relaxed);
        }
    }
}