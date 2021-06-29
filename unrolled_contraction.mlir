#contraction_accesses = [
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>,
  affine_map<(i) -> ()>
]

#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["reduction"]
}

func @conv1d_unrolled_contraction(%input : memref<16xf32>, %filter : memref<3xf32>, %output : memref<14xf32>) 
  attributes { passthrough = [["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
  %f0 = constant 0.0 : f32
  %c0 = constant 0 : index
  %v0 = vector.broadcast %f0 : f32 to vector<14xf32>
  %0 = vector.transfer_read %input[%c0], %f0 : memref<16xf32>, vector<16xf32>
  %1 = vector.transfer_read %filter[%c0], %f0 : memref<3xf32>, vector<3xf32>
  %2 = vector.extract_strided_slice %0 {offsets = [0], sizes = [3], strides = [1]} : vector<16xf32> to vector<3xf32>
  %3 = vector.contract #contraction_trait %1, %2, %f0 : vector<3xf32>, vector<3xf32> into f32
  %v1 = vector.insert %3, %v0[0] : f32 into vector<14xf32>
  %4 = vector.extract_strided_slice %0 {offsets = [1], sizes = [3], strides = [1]} : vector<16xf32> to vector<3xf32>
  %5 = vector.contract #contraction_trait %1, %4, %f0 : vector<3xf32>, vector<3xf32> into f32
  %v2 = vector.insert %5, %v1[1] : f32 into vector<14xf32>
  %6 = vector.extract_strided_slice %0 {offsets = [2], sizes = [3], strides = [1]} : vector<16xf32> to vector<3xf32>
  %7 = vector.contract #contraction_trait %1, %6, %f0 : vector<3xf32>, vector<3xf32> into f32
  %v3 = vector.insert %7, %v2[2] : f32 into vector<14xf32>
  %8 = vector.extract_strided_slice %0 {offsets = [3], sizes = [3], strides = [1]} : vector<16xf32> to vector<3xf32>
  %9 = vector.contract #contraction_trait %1, %8, %f0 : vector<3xf32>, vector<3xf32> into f32
  %v4 = vector.insert %9, %v3[3] : f32 into vector<14xf32>
  %10 = vector.extract_strided_slice %0 {offsets = [4], sizes = [3], strides = [1]} : vector<16xf32> to vector<3xf32>
  %11 = vector.contract #contraction_trait %1, %10, %f0 : vector<3xf32>, vector<3xf32> into f32
  %v5 = vector.insert %11, %v4[4] : f32 into vector<14xf32>
  %12 = vector.extract_strided_slice %0 {offsets = [5], sizes = [3], strides = [1]} : vector<16xf32> to vector<3xf32>
  %13 = vector.contract #contraction_trait %1, %12, %f0 : vector<3xf32>, vector<3xf32> into f32
  %v6 = vector.insert %13, %v5[5] : f32 into vector<14xf32>
  %14 = vector.extract_strided_slice %0 {offsets = [6], sizes = [3], strides = [1]} : vector<16xf32> to vector<3xf32>
  %15 = vector.contract #contraction_trait %1, %14, %f0 : vector<3xf32>, vector<3xf32> into f32
  %v7 = vector.insert %15, %v6[6] : f32 into vector<14xf32>
  %16 = vector.extract_strided_slice %0 {offsets = [7], sizes = [3], strides = [1]} : vector<16xf32> to vector<3xf32>
  %17 = vector.contract #contraction_trait %1, %16, %f0 : vector<3xf32>, vector<3xf32> into f32
  %v8 = vector.insert %17, %v7[7] : f32 into vector<14xf32>
  %18 = vector.extract_strided_slice %0 {offsets = [8], sizes = [3], strides = [1]} : vector<16xf32> to vector<3xf32>
  %19 = vector.contract #contraction_trait %1, %18, %f0 : vector<3xf32>, vector<3xf32> into f32
  %v9 = vector.insert %19, %v8[8] : f32 into vector<14xf32>
  %20 = vector.extract_strided_slice %0 {offsets = [9], sizes = [3], strides = [1]} : vector<16xf32> to vector<3xf32>
  %21 = vector.contract #contraction_trait %1, %20, %f0 : vector<3xf32>, vector<3xf32> into f32
  %v10 = vector.insert %21, %v9[9] : f32 into vector<14xf32>
  %22 = vector.extract_strided_slice %0 {offsets = [10], sizes = [3], strides = [1]} : vector<16xf32> to vector<3xf32>
  %23 = vector.contract #contraction_trait %1, %22, %f0 : vector<3xf32>, vector<3xf32> into f32
  %v11 = vector.insert %23, %v10[10] : f32 into vector<14xf32>
  %24 = vector.extract_strided_slice %0 {offsets = [11], sizes = [3], strides = [1]} : vector<16xf32> to vector<3xf32>
  %25 = vector.contract #contraction_trait %1, %24, %f0 : vector<3xf32>, vector<3xf32> into f32
  %v12 = vector.insert %25, %v11[11] : f32 into vector<14xf32>
  %26 = vector.extract_strided_slice %0 {offsets = [12], sizes = [3], strides = [1]} : vector<16xf32> to vector<3xf32>
  %27 = vector.contract #contraction_trait %1, %26, %f0 : vector<3xf32>, vector<3xf32> into f32
  %v13 = vector.insert %27, %v12[12] : f32 into vector<14xf32>
  %28 = vector.extract_strided_slice %0 {offsets = [13], sizes = [3], strides = [1]} : vector<16xf32> to vector<3xf32>
  %29 = vector.contract #contraction_trait %1, %28, %f0 : vector<3xf32>, vector<3xf32> into f32
  %v14 = vector.insert %29, %v13[13] : f32 into vector<14xf32>
  //vector.print %v14 : vector<14xf32>
  vector.transfer_write %v14, %output[%c0] : vector<14xf32>, memref<14xf32>
  return
}

func @print_perf(%iters: index, %total_time: f64) {
  %cF = constant 3 : index
  %cO = constant 14 : index
  %flops_per_iter = muli %cF, %cO : index
  %flops = muli %iters, %flops_per_iter : index
  %flops_i64 = index_cast %flops : index to i64
  %flops_f = sitofp %flops_i64 : i64 to f64
  %flops_per_s = divf %flops_f, %total_time : f64
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
  %iters = constant 1 : index
  %input = memref.alloc() : memref<16xf32>
  %filter = memref.alloc() : memref<3xf32>
  %output = memref.alloc() : memref<14xf32>
  memref.store %f1, %filter[%c0] : memref<3xf32>
  memref.store %f2, %filter[%c1] : memref<3xf32>
  memref.store %f3, %filter[%c2] : memref<3xf32>
  linalg.fill(%f4, %input) : f32, memref<16xf32>
  scf.for %arg0 = %c0 to %iters step %c1 {
    call @conv1d_unrolled_contraction(%input, %filter, %output) : (memref<16xf32>, memref<3xf32>, memref<14xf32>) -> ()
  }
  %t_start = call @rtclock() : () -> f64
  scf.for %arg0 = %c0 to %iters step %c1 {
    call @conv1d_unrolled_contraction(%input, %filter, %output) : (memref<16xf32>, memref<3xf32>, memref<14xf32>) -> ()
  }
  %t_end = call @rtclock() : () -> f64
  %t_conv = subf %t_end, %t_start: f64
  call @print_perf(%iters, %t_conv) : (index, f64) -> ()
  return
}

func private @rtclock() -> f64
