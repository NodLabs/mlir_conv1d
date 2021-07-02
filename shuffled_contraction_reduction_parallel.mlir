#contraction_accesses = [
  affine_map<(k, n) -> (k, n)>,
  affine_map<(k, n) -> (k)>,
  affine_map<(k, n) -> (n)>
]

#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["reduction", "parallel"]
}

func @conv1d_shuffled_contraction(%input : memref<${M}xf32>, %filter : memref<${K}xf32>, %output : memref<${N}xf32>) 
  attributes { passthrough = ["noinline", ["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %f0 = constant 0.0 : f32
  %0 = vector.transfer_read %filter[%c0], %f0 : memref<${K}xf32>, vector<${K}xf32>
  %1 = vector.transfer_read %input[%c0], %f0 : memref<${M}xf32>, vector<${M}xf32>
  // %2 = vector.shuffle %1, %1 
  //   [0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7,
  //    6, 7, 8, 7, 8, 9, 8, 9, 10, 9, 10, 11, 10, 11, 12, 11,
  //    12, 13, 12, 13, 14, 13, 14, 15] : vector<${M}xf32>, vector<${M}xf32>
  %2 = vector.shuffle %1, %1 
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
     2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<${M}xf32>, vector<${M}xf32>
  %3 = vector.shape_cast %2 : vector<42xf32> to vector<${K}x${N}xf32>
  %v0 = vector.broadcast %f0 : f32 to vector<${N}xf32>
  %4 = vector.contract #contraction_trait %3, %0, %v0 : vector<${K}x${N}xf32>, vector<${K}xf32> into vector<${N}xf32>
  // vector.print %4 : vector<${N}xf32>
  vector.transfer_write %4, %output[%c0] : vector<${N}xf32>, memref<${N}xf32>
  return
}

func @print_perf(%iters: index, %total_time: f64) {
  %cF = constant ${K} : index
  %cO = constant ${N} : index
  %flops_per_iter = muli %cF, %cO : index
  %flops = muli %iters, %flops_per_iter : index
  %flops_i64 = index_cast %flops : index to i64
  %flops_f = sitofp %flops_i64 : i64 to f64
  %flops_per_s = divf %flops_f, %total_time : f64
  vector.print %total_time : f64
  vector.print %flops_per_s : f64
  return
}

func @main() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %f1 = constant 0.02914738655090332 : f32
  %f2 = constant 0.8740115165710449 : f32
  %f3 = constant -0.858701229095459 : f32
  %f4 = constant 1.0533758 : f32
  %iters = constant ${ITERS} : index
  %input = memref.alloc() : memref<${M}xf32>
  %filter = memref.alloc() : memref<${K}xf32>
  %output = memref.alloc() : memref<${N}xf32>
  memref.store %f1, %filter[%c0] : memref<${K}xf32>
  memref.store %f2, %filter[%c1] : memref<${K}xf32>
  memref.store %f3, %filter[%c2] : memref<${K}xf32>
  linalg.fill(%f4, %input) : f32, memref<${M}xf32>
  call @conv1d_shuffled_contraction(%input, %filter, %output) : (memref<${M}xf32>, memref<${K}xf32>, memref<${N}xf32>) -> ()
  %t_start = call @rtclock() : () -> f64
  scf.for %arg0 = %c0 to %iters step %c1 {
    call @conv1d_shuffled_contraction(%input, %filter, %output) : (memref<${M}xf32>, memref<${K}xf32>, memref<${N}xf32>) -> ()
  }
  %t_end = call @rtclock() : () -> f64
  %t_conv = subf %t_end, %t_start: f64
  call @print_perf(%iters, %t_conv) : (index, f64) -> ()
  
  %p = memref.cast %output : memref<${N}xf32> to memref<*xf32>
  call @print_memref_f32(%p) : (memref<*xf32>) -> ()
  return
}

func private @rtclock() -> f64
func private @print_memref_f32(%ptr : memref<*xf32>)

