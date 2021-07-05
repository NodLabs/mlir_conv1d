#contraction_accesses = [
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>,
  affine_map<(i) -> ()>
]

#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["reduction"]
}

func @compute(%input : memref<${M}xf32>, %filter : memref<${K}xf32>, %output : memref<${N}xf32>) 
  attributes { passthrough = ["noinline", ["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
  %f0 = constant 0.0 : f32
  %c0 = constant 0 : index
  %v0 = vector.broadcast %f0 : f32 to vector<${N}xf32>
  %0 = vector.transfer_read %input[%c0], %f0 : memref<${M}xf32>, vector<${M}xf32>
  %1 = vector.transfer_read %filter[%c0], %f0 : memref<${K}xf32>, vector<${K}xf32>
  %2 = vector.extract_strided_slice %0 {offsets = [0], sizes = [3], strides = [1]} : vector<${M}xf32> to vector<${K}xf32>
  %3 = vector.contract #contraction_trait %1, %2, %f0 : vector<${K}xf32>, vector<${K}xf32> into f32
  %v1 = vector.insert %3, %v0[0] : f32 into vector<${N}xf32>
  %4 = vector.extract_strided_slice %0 {offsets = [1], sizes = [3], strides = [1]} : vector<${M}xf32> to vector<${K}xf32>
  %5 = vector.contract #contraction_trait %1, %4, %f0 : vector<${K}xf32>, vector<${K}xf32> into f32
  %v2 = vector.insert %5, %v1[1] : f32 into vector<${N}xf32>
  %6 = vector.extract_strided_slice %0 {offsets = [2], sizes = [3], strides = [1]} : vector<${M}xf32> to vector<${K}xf32>
  %7 = vector.contract #contraction_trait %1, %6, %f0 : vector<${K}xf32>, vector<${K}xf32> into f32
  %v3 = vector.insert %7, %v2[2] : f32 into vector<${N}xf32>
  %8 = vector.extract_strided_slice %0 {offsets = [3], sizes = [3], strides = [1]} : vector<${M}xf32> to vector<${K}xf32>
  %9 = vector.contract #contraction_trait %1, %8, %f0 : vector<${K}xf32>, vector<${K}xf32> into f32
  %v4 = vector.insert %9, %v3[3] : f32 into vector<${N}xf32>
  %10 = vector.extract_strided_slice %0 {offsets = [4], sizes = [3], strides = [1]} : vector<${M}xf32> to vector<${K}xf32>
  %11 = vector.contract #contraction_trait %1, %10, %f0 : vector<${K}xf32>, vector<${K}xf32> into f32
  %v5 = vector.insert %11, %v4[4] : f32 into vector<${N}xf32>
  %12 = vector.extract_strided_slice %0 {offsets = [5], sizes = [3], strides = [1]} : vector<${M}xf32> to vector<${K}xf32>
  %13 = vector.contract #contraction_trait %1, %12, %f0 : vector<${K}xf32>, vector<${K}xf32> into f32
  %v6 = vector.insert %13, %v5[5] : f32 into vector<${N}xf32>
  %14 = vector.extract_strided_slice %0 {offsets = [6], sizes = [3], strides = [1]} : vector<${M}xf32> to vector<${K}xf32>
  %15 = vector.contract #contraction_trait %1, %14, %f0 : vector<${K}xf32>, vector<${K}xf32> into f32
  %v7 = vector.insert %15, %v6[6] : f32 into vector<${N}xf32>
  %16 = vector.extract_strided_slice %0 {offsets = [7], sizes = [3], strides = [1]} : vector<${M}xf32> to vector<${K}xf32>
  %17 = vector.contract #contraction_trait %1, %16, %f0 : vector<${K}xf32>, vector<${K}xf32> into f32
  %v8 = vector.insert %17, %v7[7] : f32 into vector<${N}xf32>
  %18 = vector.extract_strided_slice %0 {offsets = [8], sizes = [3], strides = [1]} : vector<${M}xf32> to vector<${K}xf32>
  %19 = vector.contract #contraction_trait %1, %18, %f0 : vector<${K}xf32>, vector<${K}xf32> into f32
  %v9 = vector.insert %19, %v8[8] : f32 into vector<${N}xf32>
  %20 = vector.extract_strided_slice %0 {offsets = [9], sizes = [3], strides = [1]} : vector<${M}xf32> to vector<${K}xf32>
  %21 = vector.contract #contraction_trait %1, %20, %f0 : vector<${K}xf32>, vector<${K}xf32> into f32
  %v10 = vector.insert %21, %v9[9] : f32 into vector<${N}xf32>
  %22 = vector.extract_strided_slice %0 {offsets = [10], sizes = [3], strides = [1]} : vector<${M}xf32> to vector<${K}xf32>
  %23 = vector.contract #contraction_trait %1, %22, %f0 : vector<${K}xf32>, vector<${K}xf32> into f32
  %v11 = vector.insert %23, %v10[10] : f32 into vector<${N}xf32>
  %24 = vector.extract_strided_slice %0 {offsets = [11], sizes = [3], strides = [1]} : vector<${M}xf32> to vector<${K}xf32>
  %25 = vector.contract #contraction_trait %1, %24, %f0 : vector<${K}xf32>, vector<${K}xf32> into f32
  %v12 = vector.insert %25, %v11[11] : f32 into vector<${N}xf32>
  %26 = vector.extract_strided_slice %0 {offsets = [12], sizes = [3], strides = [1]} : vector<${M}xf32> to vector<${K}xf32>
  %27 = vector.contract #contraction_trait %1, %26, %f0 : vector<${K}xf32>, vector<${K}xf32> into f32
  %v13 = vector.insert %27, %v12[12] : f32 into vector<${N}xf32>
  %28 = vector.extract_strided_slice %0 {offsets = [13], sizes = [3], strides = [1]} : vector<${M}xf32> to vector<${K}xf32>
  %29 = vector.contract #contraction_trait %1, %28, %f0 : vector<${K}xf32>, vector<${K}xf32> into f32
  %v14 = vector.insert %29, %v13[13] : f32 into vector<${N}xf32>
  //vector.print %v14 : vector<${N}xf32>
  vector.transfer_write %v14, %output[%c0] : vector<${N}xf32>, memref<${N}xf32>
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
  %vInput = constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]> :
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

  // CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [14] strides = [1] data = 
  // CHECK: [8,  14,  20,  26,  32,  38,  44,  50,  56,  62,  68,  74,  80,  86]
  call @print_memref_f32(%p) : (memref<*xf32>) -> ()

  %t_conv = subf %t_end, %t_start: f64
  call @print_perf(%iters, %t_conv) : (index, f64) -> ()

  return
}

func private @rtclock() -> f64
func private @print_memref_f32(%ptr : memref<*xf32>)
