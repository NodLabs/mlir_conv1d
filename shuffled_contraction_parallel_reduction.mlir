// Tested on Intel(R) Xeon(R) CPU @ 2.00GHz w/ AVX512
func @compute(%input : memref<${M}xf32>, %filter : memref<${K}xf32>, %output : memref<${N}xf32>) 
    attributes { passthrough = ["noinline" ${TARGET_CPU} ${PREFER_VECTOR_WIDTH}]} {
  call @compute_v${FN}(%input, %filter, %output) : (memref<${M}xf32>, memref<${K}xf32>, memref<${N}xf32>) -> ()
  return
}

#contraction_accesses = [
  affine_map<(n, k) -> (n, k)>,
  affine_map<(n, k) -> (k)>,
  affine_map<(n, k) -> (n)>
]

#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "reduction"]
}

// Size 18 * 3 -> 16
// ~4.6 GFlops/s when inlined
// Iterations:        100
// Instructions:      7300
// Total Cycles:      4732
// Total uOps:        8400

// Dispatch Width:    6
// uOps Per Cycle:    1.78
// IPC:               1.54
// Block RThroughput: 46.0
func @compute_v1(%input : memref<${M}xf32>, %filter : memref<${K}xf32>, %output : memref<${N}xf32>) 
  attributes { passthrough = ["inline" ${TARGET_CPU} ${PREFER_VECTOR_WIDTH}]} {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %cf0 = constant 0.0 : f32

  // %acc = constant dense<0.0> : vector<${N}xf32>
  %acc = vector.transfer_read %output[%c0], %cf0 : memref<${N}xf32>, vector<${N}xf32>
  %rhs = vector.transfer_read %filter[%c0], %cf0 : memref<${K}xf32>, vector<${K}xf32>
  %lhs0 = vector.transfer_read %input[%c0], %cf0 : memref<${M}xf32>, vector<${M}xf32>
  %lhs1 = vector.shuffle %lhs0, %lhs0
    [0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7,
     6, 7, 8, 7, 8, 9, 8, 9, 10, 9, 10, 11, 10, 11, 12, 11,
     12, 13, 12, 13, 14, 13, 14, 15, 14, 15, 16, 15, 16, 17] : vector<${M}xf32>, vector<${M}xf32>
  %lhs = vector.shape_cast %lhs1 : vector<48xf32> to vector<${N}x${K}xf32>
  %res = vector.contract #contraction_trait %lhs, %rhs, %acc : vector<${N}x${K}xf32>, vector<${K}xf32> into vector<${N}xf32>
  vector.transfer_write %res, %output[%c0] : vector<${N}xf32>, memref<${N}xf32>
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
  %vOutput = constant dense<1.0> : vector<${N}xf32>
  memref.store %vOutput, %mvoutput[] : memref<vector<${N}xf32>>
  
  %input = vector.type_cast %mvinput: memref<vector<${M}xf32>> to memref<${M}xf32>
  %filter = vector.type_cast %mvfilter: memref<vector<${K}xf32>> to memref<${K}xf32>
  %output = vector.type_cast %mvoutput: memref<vector<${N}xf32>> to memref<${N}xf32>

  call @compute(%input, %filter, %output) : (memref<${M}xf32>, memref<${K}xf32>, memref<${N}xf32>) -> ()
  // CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [16] strides = [1] data = 
  // CHECK: [9,  15,  21,  27,  33,  39,  45,  51,  57,  63,  69,  75,  81,  87,  93,  99]
  %p = memref.cast %output : memref<${N}xf32> to memref<*xf32>
  call @print_memref_f32(%p) : (memref<*xf32>) -> ()

  %t_start = call @rtclock() : () -> f64
  scf.for %arg0 = %c0 to %iters step %c1 {
    call @compute(%input, %filter, %output) : (memref<${M}xf32>, memref<${K}xf32>, memref<${N}xf32>) -> ()
  }
  %t_end = call @rtclock() : () -> f64

  %t_conv = subf %t_end, %t_start: f64
  call @print_perf(%iters, %t_conv) : (index, f64) -> ()

  return
}

func private @rtclock() -> f64
func private @print_memref_f32(%ptr : memref<*xf32>)
