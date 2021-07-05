// Tested on Intel(R) Xeon(R) CPU @ 2.00GHz w/ AVX512
func @compute(%input : memref<${M}xf32>, %filter : memref<${K}xf32>, %output : memref<${N}xf32>) 
    attributes { passthrough = ["inline", ["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
  call @compute_v4(%input, %filter, %output) : (memref<${M}xf32>, memref<${K}xf32>, memref<${N}xf32>) -> ()
  return
}

// Size 18 * 3 -> 16
// ~35 GFlops/s when inlined
// Iterations:        100
// Instructions:      1200
// Total Cycles:      477
// Total uOps:        2400

// Dispatch Width:    6
// uOps Per Cycle:    5.03
// IPC:               2.52
// Block RThroughput: 4.5
func @compute_v1(%input : memref<${M}xf32>, %filter : memref<${K}xf32>, %output : memref<${N}xf32>) 
  attributes { passthrough = ["inline", ["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c0i32 = constant 0 : i32
  %c1i32 = constant 1 : i32
  %c2i32 = constant 2 : i32
  %cf0 = constant 0.0 : f32

  %f0 = memref.load %filter[%c0] : memref<${K}xf32>
  %f1 = memref.load %filter[%c1] : memref<${K}xf32>
  %f2 = memref.load %filter[%c2] : memref<${K}xf32>
  
  %i0 = vector.transfer_read %input[%c0], %cf0 : memref<${M}xf32>, vector<${N}xf32>
  %i1 = vector.transfer_read %input[%c1], %cf0 : memref<${M}xf32>, vector<${N}xf32>
  %i2 = vector.transfer_read %input[%c2], %cf0 : memref<${M}xf32>, vector<${N}xf32>

  %b0 = vector.broadcast %f0 : f32 to vector<${N}xf32>
  %b1 = vector.broadcast %f1 : f32 to vector<${N}xf32>
  %b2 = vector.broadcast %f2 : f32 to vector<${N}xf32>

  %acc = vector.broadcast %cf0 : f32 to vector<${N}xf32>
  %acc0 = vector.fma %i0, %b0, %acc : vector<${N}xf32>
  %acc1 = vector.fma %i1, %b1, %acc0 : vector<${N}xf32>
  %acc2 = vector.fma %i2, %b2, %acc1 : vector<${N}xf32>  
  vector.store %acc2, %output[%c0] : memref<${N}xf32>, vector<${N}xf32>

  return
}

// Size 18 * 3 -> 16
// ~35 GFlops/s when inlined
// Iterations:        100
// Instructions:      1100
// Total Cycles:      474
// Total uOps:        2300

// Dispatch Width:    6
// uOps Per Cycle:    4.85
// IPC:               2.32
// Block RThroughput: 4.5
func @compute_v2(%input : memref<${M}xf32>, %filter : memref<${K}xf32>, %output : memref<${N}xf32>) 
  attributes { passthrough = ["inline", ["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c0i32 = constant 0 : i32
  %c1i32 = constant 1 : i32
  %c2i32 = constant 2 : i32
  %cf0 = constant 0.0 : f32

  %f0 = memref.load %filter[%c0] : memref<${K}xf32>
  %f1 = memref.load %filter[%c1] : memref<${K}xf32>
  %f2 = memref.load %filter[%c2] : memref<${K}xf32>

  %i0 = vector.transfer_read %input[%c0], %cf0 : memref<${M}xf32>, vector<${N}xf32>
  %i1 = vector.transfer_read %input[%c1], %cf0 : memref<${M}xf32>, vector<${N}xf32>
  %i2 = vector.transfer_read %input[%c2], %cf0 : memref<${M}xf32>, vector<${N}xf32>

  %b0 = vector.broadcast %f0 : f32 to vector<${N}xf32>
  %b1 = vector.broadcast %f1 : f32 to vector<${N}xf32>
  %b2 = vector.broadcast %f2 : f32 to vector<${N}xf32>

  %acc0 = mulf %i0, %b0 : vector<${N}xf32>
  %acc1 = vector.fma %i1, %b1, %acc0 : vector<${N}xf32>
  %acc2 = vector.fma %i2, %b2, %acc1 : vector<${N}xf32>  
  vector.store %acc2, %output[%c0] : memref<${N}xf32>, vector<${N}xf32>

  return
}

// Size 18 * 3 -> 16
// ~35 GFlops/s when inlined
// Iterations:        100
// Instructions:      1500
// Total Cycles:      531
// Total uOps:        2600

// Dispatch Width:    6
// uOps Per Cycle:    4.90
// IPC:               2.82
// Block RThroughput: 4.5
func @compute_v3(%input : memref<${M}xf32>, %filter : memref<${K}xf32>, %output : memref<${N}xf32>) 
  attributes { passthrough = ["inline", ["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c0i32 = constant 0 : i32
  %c1i32 = constant 1 : i32
  %c2i32 = constant 2 : i32
  %cf0 = constant 0.0 : f32

  %f = vector.transfer_read %filter[%c0], %cf0 : memref<${K}xf32>, vector<${K}xf32>
  %f0 = vector.extractelement %f[%c0i32 : i32] : vector<${K}xf32>
  %f1 = vector.extractelement %f[%c1i32 : i32] : vector<${K}xf32>
  %f2 = vector.extractelement %f[%c2i32 : i32] : vector<${K}xf32>
  
  %i0 = vector.transfer_read %input[%c0], %cf0 : memref<${M}xf32>, vector<${N}xf32>
  %i1 = vector.transfer_read %input[%c1], %cf0 : memref<${M}xf32>, vector<${N}xf32>
  %i2 = vector.transfer_read %input[%c2], %cf0 : memref<${M}xf32>, vector<${N}xf32>

  %b0 = vector.broadcast %f0 : f32 to vector<${N}xf32>
  %b1 = vector.broadcast %f1 : f32 to vector<${N}xf32>
  %b2 = vector.broadcast %f2 : f32 to vector<${N}xf32>

  %acc = vector.broadcast %cf0 : f32 to vector<${N}xf32>
  %acc0 = vector.fma %i0, %b0, %acc : vector<${N}xf32>
  %acc1 = vector.fma %i1, %b1, %acc0 : vector<${N}xf32>
  %acc2 = vector.fma %i2, %b2, %acc1 : vector<${N}xf32>  
  vector.store %acc2, %output[%c0] : memref<${N}xf32>, vector<${N}xf32>

  return
}

// Size 18 * 3 -> 16
// ~42 GFlops/s when inlined
// Iterations:        100
// Instructions:      2300
// Total Cycles:      628
// Total uOps:        3400

// Dispatch Width:    6
// uOps Per Cycle:    5.41
// IPC:               3.66
// Block RThroughput: 5.7
func @compute_v4(%input : memref<${M}xf32>, %filter : memref<${K}xf32>, %output : memref<${N}xf32>) 
  attributes { passthrough = ["inline", ["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %cf0 = constant 0.0 : f32

  %in0 = vector.transfer_read %input[%c0], %cf0 : memref<${M}xf32>, vector<${M}xf32>
  %f = vector.transfer_read %filter[%c0], %cf0 : memref<${K}xf32>, vector<${K}xf32>

  %in1 = vector.shuffle %in0, %in0
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
     2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] : vector<${M}xf32>, vector<${M}xf32>
  %in = vector.shape_cast %in1 : vector<48xf32> to vector<${K}x${N}xf32>

  %6 = vector.broadcast %f : vector<${K}xf32> to vector<${N}x${K}xf32>
  %7 = vector.transpose %6, [1, 0] : vector<${N}x${K}xf32> to vector<${K}x${N}xf32>
  // vector.print %7 : vector<${K}x${N}xf32>

  // %v5 = vector.multi_reduction #vector.kind<mul>, %v4 [0] : vector<${2x${K}x${N}xf32> to vector<${K}x${N}xf32>
  %v5 = mulf %in, %7 : vector<${K}x${N}xf32>

  // %v6 = vector.multi_reduction #vector.kind<add>, %v5 [0] : vector<${K}x${N}xf32> to vector<${N}xf32>
  %ve0 = vector.extract %v5[0] : vector<${K}x${N}xf32>
  %ve1 = vector.extract %v5[1] : vector<${K}x${N}xf32>
  %ve2 = vector.extract %v5[2] : vector<${K}x${N}xf32>
  %res0 = addf %ve0, %ve1 : vector<${N}xf32>
  %res1 = addf %res0, %ve2 : vector<${N}xf32>
  
  vector.transfer_write %res1, %output[%c0] : vector<${N}xf32> , memref<${N}xf32>

  return
}

// Size 18 * 3 -> 16
// ~43 GFlops/s when inlined
// Iterations:        100
// Instructions:      1800
// Total Cycles:      533
// Total uOps:        3000

// Dispatch Width:    6
// uOps Per Cycle:    5.63
// IPC:               3.38
// Block RThroughput: 5.0
func @compute_v5(%input : memref<${M}xf32>, %filter : memref<${K}xf32>, %output : memref<${N}xf32>) 
  attributes { passthrough = ["inline", ["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %cf0 = constant 0.0 : f32

  %in0 = vector.transfer_read %input[%c0], %cf0 : memref<${M}xf32>, vector<${M}xf32>

  %f0 = memref.load %filter[%c0] : memref<${K}xf32>
  %f1 = memref.load %filter[%c1] : memref<${K}xf32>
  %f2 = memref.load %filter[%c2] : memref<${K}xf32>
  %b = vector.broadcast %cf0 : f32 to vector<${K}xf32>
  %b0 = vector.insert %f0, %b[0] : f32 into vector<${K}xf32>
  %b1 = vector.insert %f1, %b0[1] : f32 into vector<${K}xf32>
  %f = vector.insert %f2, %b1[2] : f32 into vector<${K}xf32>
  
  %in1 = vector.shuffle %in0, %in0
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
     2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] : vector<${M}xf32>, vector<${M}xf32>
  %in = vector.shape_cast %in1 : vector<48xf32> to vector<${K}x${N}xf32>

  %6 = vector.broadcast %f : vector<${K}xf32> to vector<${N}x${K}xf32>
  %7 = vector.transpose %6, [1, 0] : vector<${N}x${K}xf32> to vector<${K}x${N}xf32>

  // %v5 = vector.multi_reduction #vector.kind<mul>, %v4 [0] : vector<2x${K}x${N}xf32> to vector<${K}x${N}xf32>
  %v5 = mulf %in, %7 : vector<${K}x${N}xf32>

  // %v6 = vector.multi_reduction #vector.kind<add>, %v5 [0] : vector<${K}x${N}xf32> to vector<${N}xf32>
  %ve0 = vector.extract %v5[0] : vector<${K}x${N}xf32>
  %ve1 = vector.extract %v5[1] : vector<${K}x${N}xf32>
  %ve2 = vector.extract %v5[2] : vector<${K}x${N}xf32>
  %res0 = addf %ve0, %ve1 : vector<${N}xf32>
  %res1 = addf %res0, %ve2 : vector<${N}xf32>
  
  vector.transfer_write %res1, %output[%c0] : vector<${N}xf32> , memref<${N}xf32>

  return
}

// Size 18 * 3 -> 16
// ~43 GFlops/s when inlined
// Iterations:        100
// Instructions:      2300
// Total Cycles:      628
// Total uOps:        3400

// Dispatch Width:    6
// uOps Per Cycle:    5.41
// IPC:               3.66
// Block RThroughput: 5.7
func @compute_v6(%input : memref<${M}xf32>, %filter : memref<${K}xf32>, %output : memref<${N}xf32>) 
  attributes { passthrough = ["inline", ["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %cf0 = constant 0.0 : f32

  %f = vector.transfer_read %filter[%c0], %cf0 : memref<${K}xf32>, vector<${K}xf32>
  %in0 = vector.transfer_read %input[%c0], %cf0 : memref<${M}xf32>, vector<${M}xf32>

  %in1 = vector.shuffle %in0, %in0
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
     2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] : vector<${M}xf32>, vector<${M}xf32>
  %in = vector.shape_cast %in1 : vector<48xf32> to vector<${K}x${N}xf32>

  %6 = vector.broadcast %f : vector<${K}xf32> to vector<${N}x${K}xf32>
  %7 = vector.transpose %6, [1, 0] : vector<${N}x${K}xf32> to vector<${K}x${N}xf32>

  // %v5 = vector.multi_reduction #vector.kind<mul>, %v4 [0] : vector<2x${K}x${N}xf32> to vector<${K}x${N}xf32>
  %v5 = mulf %in, %7 : vector<${K}x${N}xf32>

  // %v6 = vector.multi_reduction #vector.kind<add>, %v5 [0] : vector<${K}x${N}xf32> to vector<${N}xf32>
  %ve0 = vector.extract %v5[0] : vector<${K}x${N}xf32>
  %ve1 = vector.extract %v5[1] : vector<${K}x${N}xf32>
  %ve2 = vector.extract %v5[2] : vector<${K}x${N}xf32>
  %res0 = addf %ve0, %ve1 : vector<${N}xf32>
  %res1 = addf %res0, %ve2 : vector<${N}xf32>

  vector.transfer_write %res1, %output[%c0] : vector<${N}xf32> , memref<${N}xf32>

  return
}

// Size 18 * 3 -> 16
// ~3 GFlops/s when inlined
// Iterations:        100
// Instructions:      12500
// Total Cycles:      6230
// Total uOps:        13700

// Dispatch Width:    6
// uOps Per Cycle:    2.20
// IPC:               2.01
// Block RThroughput: 62.0
func @compute_v7(%input : memref<${M}xf32>, %filter : memref<${K}xf32>, %output : memref<${N}xf32>) 
  attributes { passthrough = ["inline", ["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %cf0 = constant 0.0 : f32

  %f = vector.transfer_read %filter[%c0], %cf0 : memref<${K}xf32>, vector<${K}xf32>
  %in0 = vector.transfer_read %input[%c0], %cf0 : memref<${M}xf32>, vector<${M}xf32>

  %in1 = vector.shuffle %in0, %in0
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
     2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] : vector<${M}xf32>, vector<${M}xf32>
  %in = vector.shape_cast %in1 : vector<48xf32> to vector<${K}x${N}xf32>

  %6 = vector.broadcast %f : vector<${K}xf32> to vector<${N}x${K}xf32>
  %7 = vector.transpose %6, [1, 0] : vector<${N}x${K}xf32> to vector<${K}x${N}xf32>

  // %v5 = vector.multi_reduction #vector.kind<mul>, %v4 [0] : vector<2x${K}x${N}xf32> to vector<${K}x${N}xf32>
  %v5 = mulf %in, %7 : vector<${K}x${N}xf32>

  %res1 = vector.multi_reduction #vector.kind<add>, %v5 [0] : vector<${K}x${N}xf32> to vector<${N}xf32>
  vector.transfer_write %res1, %output[%c0] : vector<${N}xf32> , memref<${N}xf32>

  return
}

// This one is so bad it does not want to inline even when told so.
// Size 18 * 3 -> 16
// ~1.9 GFlops/s
// Manually extracting assembly fomr compute_v8 yields:
// Iterations:        100
// Instructions:      18900
// Total Cycles:      9123
// Total uOps:        24100

// Dispatch Width:    6
// uOps Per Cycle:    2.64
// IPC:               2.07
// Block RThroughput: 67.0
func @compute_v8(%input : memref<${M}xf32>, %filter : memref<${K}xf32>, %output : memref<${N}xf32>) 
  attributes { passthrough = ["inline", ["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %f0 = constant 0.0 : f32

  %z0 = vector.broadcast %f0 : f32 to vector<${M}xf32>
  %v0 = vector.broadcast %f0 : f32 to vector<2x${K}x${M}xf32>
  %1 = vector.transfer_read %filter[%c0], %f0 : memref<${K}xf32>, vector<${K}xf32>
  
  %2 = vector.constant_mask [${N}] : vector<${M}xi1>

  %3 = vector.expandload %input[%c0], %2, %z0 : memref<${M}xf32>, vector<${M}xi1>, vector<${M}xf32> into vector<${M}xf32>
  %v1 = vector.insert %3, %v0[0, 0] : vector<${M}xf32> into vector<2x${K}x${M}xf32>

  %4 = vector.expandload %input[%c1], %2, %z0 : memref<${M}xf32>, vector<${M}xi1>, vector<${M}xf32> into vector<${M}xf32>
  %v2 = vector.insert %4, %v1[0, 1] : vector<${M}xf32> into vector<2x${K}x${M}xf32>

  %5 = vector.expandload %input[%c2], %2, %z0 : memref<${M}xf32>, vector<${M}xi1>, vector<${M}xf32> into vector<${M}xf32>
  %v3 = vector.insert %5, %v2[0, 2] : vector<${M}xf32> into vector<2x${K}x${M}xf32>

  %6 = vector.broadcast %1 : vector<${K}xf32> to vector<${M}x${K}xf32>
  %7 = vector.transpose %6, [1, 0] : vector<${M}x${K}xf32> to vector<${K}x${M}xf32>
  %v4 = vector.insert %7, %v3[1] : vector<${K}x${M}xf32> into vector<2x${K}x${M}xf32>

  %v5 = vector.multi_reduction #vector.kind<mul>, %v4 [0] : vector<2x${K}x${M}xf32> to vector<${K}x${M}xf32>
  %v6 = vector.multi_reduction #vector.kind<add>, %v5 [0] : vector<${K}x${M}xf32> to vector<${M}xf32>

  %v7 = vector.extract_strided_slice %v6 {offsets = [0], sizes=[${N}], strides=[1]} : vector<${M}xf32> to vector<${N}xf32>
  
  vector.transfer_write %v7, %output[%c0] : vector<${N}xf32> , memref<${N}xf32>

  return
}

func @print_perf(%iters: index, %total_time: f64) {
  %c1 = constant 1: index
  %cF = constant ${K} : index
  %cO = constant ${N} : index

  // For each output point we have ${K} muls and ${K-1} adds.
  %cFm1 = subi %cF, %c1 : index
  %flops_filter = addi %cF, %cFm1 : index
  %flops_per_iter = muli %cO, %flops_filter : index

  %flops = muli %iters, %flops_per_iter : index
  %flops_i64 = index_cast %flops : index to i64
  %flops_f = sitofp %flops_i64 : i64 to f64
  %flops_per_s = divf %flops_f, %total_time : f64
  // vector.print %total_time : f64
  vector.print %flops_per_s : f64
  return
}

func @main() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %iters = constant ${ITERS} : index

  %mvinput = memref.alloc() : memref<vector<${M}xf32>>
  %vInput = constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]> :
    vector<${M}xf32>
  memref.store %vInput, %mvinput[] : memref<vector<${M}xf32>>

  %mvfilter = memref.alloc() : memref<vector<${K}xf32>>
  %vFilter = constant dense<[1.0, 2.0, 3.0]> : vector<${K}xf32>
  memref.store %vFilter, %mvfilter[] : memref<vector<${K}xf32>>

  %mvoutput = memref.alloc() : memref<vector<${N}xf32>>
  %vOutput = constant dense<0.0> : vector<${N}xf32>
  memref.store %vOutput, %mvoutput[] : memref<vector<${N}xf32>>
  
  %input = vector.type_cast %mvinput: memref<vector<${M}xf32>> to memref<${M}xf32>
  %filter = vector.type_cast %mvfilter: memref<vector<${K}xf32>> to memref<${K}xf32>
  %output = vector.type_cast %mvoutput: memref<vector<${N}xf32>> to memref<${N}xf32>
  call @compute(%input, %filter, %output) : (memref<${M}xf32>, memref<${K}xf32>, memref<${N}xf32>) -> ()
  %t_start = call @rtclock() : () -> f64
  scf.for %arg0 = %c0 to %iters step %c1 {
    call @compute(%input, %filter, %output) : (memref<${M}xf32>, memref<${K}xf32>, memref<${N}xf32>) -> ()
  }
  %t_end = call @rtclock() : () -> f64
  
  %p = memref.cast %output : memref<${N}xf32> to memref<*xf32>

  // CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [16] strides = [1] data = 
  // CHECK: [8,  14,  20,  26,  32,  38,  44,  50,  56,  62,  68,  74,  80,  86,  92,  98]
  call @print_memref_f32(%p) : (memref<*xf32>) -> ()

  %t_conv = subf %t_end, %t_start: f64
  call @print_perf(%iters, %t_conv) : (index, f64) -> ()

  return
}

func private @rtclock() -> f64
func private @print_memref_f32(%ptr : memref<*xf32>)
