//
//  mps.metal
//  mps
//
//  Created by Sidharth Niwas Sharma on 12/1/25.
//

#include <metal_stdlib>
#include <metal_atomic>

using namespace metal;

// Arguments:
// - arrayA, arrayB, arrayC, arrayD: The buffer data passed from the host application.
// - gid: The thread position/index within the grid, provided by the GPU.

kernel void fma_test(
                       const device float* arrayA [[buffer(0)]],
                       const device float* arrayB [[buffer(1)]],
                       const device float* arrayC [[buffer(2)]],
                       device float* arrayD [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]
                       )
{
    arrayD[gid] = fma(arrayA[gid], arrayB[gid], arrayC[gid]);
}

kernel void sparse_read_test(
                             const device uint* indices [[buffer(0)]],
                             const device float* values [[buffer(1)]],
                             device float* output_pointer [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]
                             )
{
    float value_read = values[indices[gid]];
    device atomic_float* atomic_output_pointer = (device atomic_float*) output_pointer;
    metal::atomic_fetch_add_explicit(atomic_output_pointer, value_read, metal::memory_order_relaxed);
    
}





