//
//  main.swift
//  mps
//
//  Created by Sidharth Niwas Sharma on 12/1/25.
//

import Foundation
import Metal

// 1. Configuration
let arrayLength = 10_000_000 // 10 million elements
let indicesLength = 100_000
let iterations = 10          // Run multiple times to average out overhead

// Helper to generate random data
func generateRandomArray(count: Int) -> [Float] {
    return (0..<count).map { _ in Float.random(in: 0...100) }
}

func generateRandomIntArray(count: Int) -> [UInt32] {
    return (0..<count).map { _ in UInt32.random(in: 0...UInt32(arrayLength))}
}

// Helper for benchmarking
func benchmark(name: String, block: () -> Void) {
    let start = CFAbsoluteTimeGetCurrent()
    block()
    let end = CFAbsoluteTimeGetCurrent()
    print("\(name): \(String(format: "%.5f", end - start)) seconds")
}

func main() {
    print("Starting Metal Benchmark")
    print("Array Size: \(arrayLength) elements")
    print("Memory Size: \(arrayLength * MemoryLayout<Float>.size / 1024 / 1024) MB per array")

    // 2. Metal Setup
    guard let device = MTLCreateSystemDefaultDevice() else {
        fatalError("GPU not available")
    }
    print("GPU: \(device.name)")

    guard let library = device.makeDefaultLibrary() else {
        fatalError("Could not load default library. Did you add fma_sparse.metal to the target?")
    }

    guard let function = library.makeFunction(name: "fma_test") else {
        fatalError("Could not find function 'fma_test' in fma_sparse.metal")
    }

    let pipelineState: MTLComputePipelineState
    do {
        pipelineState = try device.makeComputePipelineState(function: function)
    } catch {
        fatalError("Failed to create pipeline state: \(error)")
    }

    guard let commandQueue = device.makeCommandQueue() else {
        fatalError("Could not create command queue")
    }

    // 3. Data Preparation
    print("\ngenerating data...")
    let arrayA = generateRandomArray(count: arrayLength)
    let arrayB = generateRandomArray(count: arrayLength)
    let arrayC = generateRandomArray(count: arrayLength)

    // Calculate buffer size in bytes
    let bufferSize = arrayLength * MemoryLayout<Float>.size

    // Create Metal buffers
    // usage: .storageModeShared allows CPU and GPU to access it (good for unified memory like M1/M2/M3)
    guard let bufferA = device.makeBuffer(bytes: arrayA, length: bufferSize, options: .storageModeShared),
          let bufferB = device.makeBuffer(bytes: arrayB, length: bufferSize, options: .storageModeShared),
          let bufferC = device.makeBuffer(bytes: arrayC, length: bufferSize, options: .storageModeShared),
          let bufferD = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
        fatalError("Failed to create buffers")
    }

    // 4. Run Benchmark
    print("\nRunning GPU Benchmark (\(iterations) iterations)...")
    
    // Warmup run (shaders compilation etc often slows down the first run)
    let commandBufferWarmup = commandQueue.makeCommandBuffer()
    let encoderWarmup = commandBufferWarmup?.makeComputeCommandEncoder()
    encoderWarmup?.setComputePipelineState(pipelineState)
    encoderWarmup?.setBuffer(bufferA, offset: 0, index: 0)
    encoderWarmup?.setBuffer(bufferB, offset: 0, index: 1)
    encoderWarmup?.setBuffer(bufferC, offset: 0, index: 2)
    encoderWarmup?.setBuffer(bufferD, offset: 0, index: 3)
    let gridSizeWarmup = MTLSizeMake(arrayLength, 1, 1)
    let threadGroupSizeWarmup = MTLSizeMake(min(pipelineState.maxTotalThreadsPerThreadgroup, arrayLength), 1, 1)
    encoderWarmup?.dispatchThreads(gridSizeWarmup, threadsPerThreadgroup: threadGroupSizeWarmup)
    encoderWarmup?.endEncoding()
    commandBufferWarmup?.commit()
    commandBufferWarmup?.waitUntilCompleted()

    var totalGPUTime: Double = 0

    for i in 1...iterations {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Failed to create command buffer/encoder")
        }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(bufferC, offset: 0, index: 2)
        encoder.setBuffer(bufferD, offset: 0, index: 3)

        // Calculate grid size
        let gridSize = MTLSizeMake(arrayLength, 1, 1)
        
        // Calculate thread group size
        // We ensure we don't exceed the GPU's max threads per group
        var threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup
        if threadGroupSize > arrayLength {
            threadGroupSize = arrayLength
        }
        let threadGroupSizeStruct = MTLSizeMake(threadGroupSize, 1, 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSizeStruct)
        encoder.endEncoding()

        commandBuffer.commit()
        
        // Wait for GPU to finish to measure accurate timing
        commandBuffer.waitUntilCompleted()
        
        let endTime = CFAbsoluteTimeGetCurrent()
        let duration = endTime - startTime
        totalGPUTime += duration
        print("Run \(i): \(String(format: "%.5f", duration))s")
    }

    print("Average GPU Time: \(String(format: "%.5f", totalGPUTime / Double(iterations)))s")


    // 5. Verification (Optional but recommended)
    // Read back results
    let pointerD = bufferD.contents().bindMemory(to: Float.self, capacity: arrayLength)
    // Check just the first few
    if pointerD[0] == (arrayA[0] * arrayB[0]) + arrayC[0] {
        print("\nVerification Passed (First element matches)")
    } else {
        print("\nVerification Failed")
        print("Expected: \(arrayA[0] + arrayB[0])")
        print("Got: \(pointerD[0])")
    }
    
    
    // 6. CPU Comparison
    print("\nRunning CPU Benchmark (Single Core)...")
    let startCPU = CFAbsoluteTimeGetCurrent()
    var cpuResult = [Float](repeating: 0, count: arrayLength)
    for i in 0..<arrayLength {
        cpuResult[i] = arrayA[i] * arrayB[i] + arrayC[i]
    }
    let endCPU = CFAbsoluteTimeGetCurrent()
    print("CPU Time: \(String(format: "%.5f", endCPU - startCPU))s")
    
    
    
    
    
    guard let function2 = library.makeFunction(name: "sparse_read_test") else {
        fatalError("Could not find function 'sparse_read_test' in fma_sparse.metal")
    }

    let pipelineState2: MTLComputePipelineState
    do {
        pipelineState2 = try device.makeComputePipelineState(function: function2)
    } catch {
        fatalError("Failed to create pipeline state: \(error)")
    }

    guard let commandQueue2 = device.makeCommandQueue() else {
        fatalError("Could not create command queue")
    }

    print("\ngenerating data for the sparse test...")
    let indices = generateRandomIntArray(count: indicesLength)
    let values = generateRandomArray(count: arrayLength)
    var output_sum_val = 0.0 as Float32;

    let bufferSize_indices = indicesLength * MemoryLayout<UInt32>.size
    let bufferSize_values = arrayLength * MemoryLayout<Float>.size
    let bufferSize_output = MemoryLayout<Float>.size
    
    guard let buffer_indices = device.makeBuffer(bytes: indices, length: bufferSize_indices, options: .storageModeShared),
          let buffer_values = device.makeBuffer(bytes: values, length: bufferSize_values, options: .storageModeShared),
    let buffer_output = device.makeBuffer(bytes: &output_sum_val, length: bufferSize_output, options: .storageModeShared)
    else {
        fatalError("Failed to create buffers")
    }
    
    
    // 4. Run Benchmark
    print("\nRunning GPU Benchmark (\(iterations) iterations)...")
    
    // Warmup run (shaders compilation etc often slows down the first run)
    let commandBufferWarmup2 = commandQueue2.makeCommandBuffer()
    let encoderWarmup2 = commandBufferWarmup2?.makeComputeCommandEncoder()
    encoderWarmup2?.setComputePipelineState(pipelineState2)
    encoderWarmup2?.setBuffer(buffer_indices, offset: 0, index: 0)
    encoderWarmup2?.setBuffer(buffer_values, offset: 0, index: 1)
    encoderWarmup2?.setBuffer(buffer_output, offset: 0, index: 2)
    let gridSizeWarmup2 = MTLSizeMake(indicesLength, 1, 1)
    let threadGroupSizeWarmup2 = MTLSizeMake(min(pipelineState2.maxTotalThreadsPerThreadgroup, indicesLength), 1, 1)
    encoderWarmup2?.dispatchThreads(gridSizeWarmup2, threadsPerThreadgroup: threadGroupSizeWarmup2)
    encoderWarmup2?.endEncoding()
    commandBufferWarmup2?.commit()
    commandBufferWarmup2?.waitUntilCompleted()

    var totalGPUTime2: Double = 0

    for i in 1...iterations {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        guard let commandBuffer = commandQueue2.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Failed to create command buffer/encoder")
        }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(buffer_indices, offset: 0, index: 0)
        encoder.setBuffer(buffer_values, offset: 0, index: 1)
        encoder.setBuffer(buffer_output, offset: 0, index: 2)

        // Calculate grid size
        let gridSize = MTLSizeMake(indicesLength, 1, 1)
        
        // Calculate thread group size
        // We ensure we don't exceed the GPU's max threads per group
        var threadGroupSize = pipelineState2.maxTotalThreadsPerThreadgroup
        if threadGroupSize > indicesLength {
            threadGroupSize = indicesLength
        }
        let threadGroupSizeStruct = MTLSizeMake(threadGroupSize, 1, 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSizeStruct)
        encoder.endEncoding()

        commandBuffer.commit()
        
        // Wait for GPU to finish to measure accurate timing
        commandBuffer.waitUntilCompleted()
        
        let endTime = CFAbsoluteTimeGetCurrent()
        let duration = endTime - startTime
        totalGPUTime2 += duration
        print("Run \(i): \(String(format: "%.5f", duration))s")
    }

    print("Average GPU Time: \(String(format: "%.5f", totalGPUTime2 / Double(iterations)))s")
        
    print("\nRunning CPU Benchmark (Single Core)...")
    let startCPU2 = CFAbsoluteTimeGetCurrent()
    var cpuResult2 = 0.0 as Float32
    for i in 0..<UInt32(indicesLength) {
        cpuResult2 += values[Int(indices[Int(i)])];
    }
    let endCPU2 = CFAbsoluteTimeGetCurrent()
    print("CPU Time: \(String(format: "%.5f", endCPU2 - startCPU2))s")
    
    let outputPointer = buffer_output.contents().assumingMemoryBound(to: Float32.self)
    output_sum_val = outputPointer.pointee
    
    if (cpuResult2 - output_sum_val)/cpuResult2 < 1.0e-5 {
        print("\nPASSED")
    } else {
        print("\nFAILED")
        print("Expected: \(cpuResult2)")
        print("Got: \(output_sum_val)")
    }
}

// Execute
main()
