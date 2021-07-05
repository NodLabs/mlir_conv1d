func @compute(%input : memref<${M}xf32>, %filter : memref<${K}xf32>, %output : memref<${N}xf32>) 
  attributes { passthrough = ["noinline", ["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %f0 = constant 0.0 : f32

  %z0 = vector.broadcast %f0 : f32 to vector<${M}xf32>
  %v0 = vector.broadcast %f0 : f32 to vector<2x${K}x${M}xf32>
  %1 = vector.transfer_read %filter[%c0], %f0 : memref<${K}xf32>, vector<${K}xf32>
  // vector.print %1 : vector<${K}xf32>
  
  %2 = vector.constant_mask [${N}] : vector<${M}xi1>

  %3 = vector.expandload %input[%c0], %2, %z0 : memref<${M}xf32>, vector<${M}xi1>, vector<${M}xf32> into vector<${M}xf32>
  %v1 = vector.insert %3, %v0[0, 0] : vector<${M}xf32> into vector<2x${K}x${M}xf32>
  // vector.print %3 : vector<${M}xf32>
  // vector.print %v1 : vector<2x${K}x${M}xf32>

  %4 = vector.expandload %input[%c1], %2, %z0 : memref<${M}xf32>, vector<${M}xi1>, vector<${M}xf32> into vector<${M}xf32>
  %v2 = vector.insert %4, %v1[0, 1] : vector<${M}xf32> into vector<2x${K}x${M}xf32>
  // vector.print %4 : vector<${M}xf32>
  // vector.print %v2 : vector<2x${K}x${M}xf32>

  %5 = vector.expandload %input[%c2], %2, %z0 : memref<${M}xf32>, vector<${M}xi1>, vector<${M}xf32> into vector<${M}xf32>
  %v3 = vector.insert %5, %v2[0, 2] : vector<${M}xf32> into vector<2x${K}x${M}xf32>
  // vector.print %5 : vector<${M}xf32>
  // vector.print %v3 : vector<2x${K}x${M}xf32>

  %6 = vector.broadcast %1 : vector<${K}xf32> to vector<${M}x${K}xf32>
  %7 = vector.transpose %6, [1, 0] : vector<${M}x${K}xf32> to vector<${K}x${M}xf32>
  %v4 = vector.insert %7, %v3[1] : vector<${K}x${M}xf32> into vector<2x${K}x${M}xf32>
  // vector.print %v4 : vector<2x${K}x${M}xf32>

  %v5 = vector.multi_reduction #vector.kind<mul>, %v4 [0] : vector<2x${K}x${M}xf32> to vector<${K}x${M}xf32>
  %v6 = vector.multi_reduction #vector.kind<add>, %v5 [0] : vector<${K}x${M}xf32> to vector<${M}xf32>

  %v7 = vector.extract_strided_slice %v6 {offsets = [0], sizes=[${N}], strides=[1]} : vector<${M}xf32> to vector<${N}xf32>
  //vector.print %v7 : vector<${N}xf32>
  
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
