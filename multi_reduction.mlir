func @conv1d_multi_reduction(%input : memref<16xf32>, %filter : memref<3xf32>, %output : memref<14xf32>) 
  attributes { passthrough = [["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %f0 = constant 0.0 : f32
  %z0 = vector.broadcast %f0 : f32 to vector<16xf32>
  %v0 = vector.broadcast %f0 : f32 to vector<2x3x16xf32>
  %1 = vector.transfer_read %filter[%c0], %f0 : memref<3xf32>, vector<3xf32>
  %2 = vector.constant_mask [14] : vector<16xi1>
  %3 = vector.expandload %input[%c0], %2, %z0 : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  %v1 = vector.insert %3, %v0[0, 0] : vector<16xf32> into vector<2x3x16xf32>
  %4 = vector.expandload %input[%c1], %2, %z0 : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  %v2 = vector.insert %3, %v1[0, 1] : vector<16xf32> into vector<2x3x16xf32>
  %5 = vector.expandload %input[%c2], %2, %z0 : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  %v3 = vector.insert %3, %v2[0, 2] : vector<16xf32> into vector<2x3x16xf32>
  %6 = vector.broadcast %1 : vector<3xf32> to vector<16x3xf32>
  %7 = vector.transpose %6, [1, 0] : vector<16x3xf32> to vector<3x16xf32>
  %v4 = vector.insert %7, %v3[1] : vector<3x16xf32> into vector<2x3x16xf32>
  %v5 = vector.multi_reduction #vector.kind<mul>, %v4 [0] : vector<2x3x16xf32> to vector<3x16xf32>
  %v6 = vector.multi_reduction #vector.kind<add>, %v5 [0] : vector<3x16xf32> to vector<16xf32>
  %v7 = vector.extract_strided_slice %v6 {offsets = [0], sizes=[14], strides=[1]} : vector<16xf32> to vector<14xf32>
  //vector.print %v7 : vector<14xf32>
  vector.transfer_write %v7, %output[%c0] : vector<14xf32> , memref<14xf32>
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
    call @conv1d_multi_reduction(%input, %filter, %output) : (memref<16xf32>, memref<3xf32>, memref<14xf32>) -> ()
  }
  %t_start = call @rtclock() : () -> f64
  scf.for %arg0 = %c0 to %iters step %c1 {
    call @conv1d_multi_reduction(%input, %filter, %output) : (memref<16xf32>, memref<3xf32>, memref<14xf32>) -> ()
  }
  %t_end = call @rtclock() : () -> f64
  %t_conv = subf %t_end, %t_start: f64
  call @print_perf(%iters, %t_conv) : (index, f64) -> ()
  return
}

func private @rtclock() -> f64
