func @conv1d_scalar_unrolled(%input : memref<${M}xf32>, %filter : memref<${K}xf32>, %output : memref<${N}xf32>) 
  attributes { passthrough = ["noinline", ["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant ${K} : index
  %c4 = constant 4 : index
  %c5 = constant 5 : index
  %c6 = constant 6 : index
  %c7 = constant 7 : index
  %c8 = constant 8 : index
  %c9 = constant 9 : index
  %c10 = constant 10 : index
  %c11 = constant 11 : index
  %c12 = constant 12 : index
  %c13 = constant 13 : index
  %c14 = constant ${N} : index
  %f0 = constant 0.0 : f32
  %x0 = scf.for %j = %c0 to %c3 step %c1 iter_args(%acc = %f0) -> (f32) {
    %start_idx = constant 0 : index
    %idx = addi %start_idx, %j : index
    %0 = memref.load %input[%idx] : memref<${M}xf32>
    %1 = memref.load %filter[%j] : memref<${K}xf32>
    %3 = mulf %0, %1 : f32
    %4 = addf %acc, %3 : f32
    scf.yield %4 : f32
  }
  memref.store %x0, %output[%c0] : memref<${N}xf32>
  //vector.print %x0 : f32
  %x1 = scf.for %j = %c0 to %c3 step %c1 iter_args(%acc = %f0) -> (f32) {
    %start_idx = constant 1 : index
    %idx = addi %start_idx, %j : index
    %0 = memref.load %input[%idx] : memref<${M}xf32>
    %1 = memref.load %filter[%j] : memref<${K}xf32>
    %3 = mulf %0, %1 : f32
    %4 = addf %acc, %3 : f32
    scf.yield %4 : f32
  }
  memref.store %x1, %output[%c1] : memref<${N}xf32>
  //vector.print %x1 : f32
  %x2 = scf.for %j = %c0 to %c3 step %c1 iter_args(%acc = %f0) -> (f32) {
    %start_idx = constant 2 : index
    %idx = addi %start_idx, %j : index
    %0 = memref.load %input[%idx] : memref<${M}xf32>
    %1 = memref.load %filter[%j] : memref<${K}xf32>
    %3 = mulf %0, %1 : f32
    %4 = addf %acc, %3 : f32
    scf.yield %4 : f32
  }
  memref.store %x2, %output[%c2] : memref<${N}xf32>
  //vector.print %x2 : f32
  %x3 = scf.for %j = %c0 to %c3 step %c1 iter_args(%acc = %f0) -> (f32) {
    %start_idx = constant ${K} : index
    %idx = addi %start_idx, %j : index
    %0 = memref.load %input[%idx] : memref<${M}xf32>
    %1 = memref.load %filter[%j] : memref<${K}xf32>
    %3 = mulf %0, %1 : f32
    %4 = addf %acc, %3 : f32
    scf.yield %4 : f32
  }
  memref.store %x3, %output[%c3] : memref<${N}xf32>
  //vector.print %x3 : f32
  %x4 = scf.for %j = %c0 to %c3 step %c1 iter_args(%acc = %f0) -> (f32) {
    %start_idx = constant 4 : index
    %idx = addi %start_idx, %j : index
    %0 = memref.load %input[%idx] : memref<${M}xf32>
    %1 = memref.load %filter[%j] : memref<${K}xf32>
    %3 = mulf %0, %1 : f32
    %4 = addf %acc, %3 : f32
    scf.yield %4 : f32
  }
  memref.store %x4, %output[%c4] : memref<${N}xf32>
  //vector.print %x4 : f32
  %x5 = scf.for %j = %c0 to %c3 step %c1 iter_args(%acc = %f0) -> (f32) {
    %start_idx = constant 5 : index
    %idx = addi %start_idx, %j : index
    %0 = memref.load %input[%idx] : memref<${M}xf32>
    %1 = memref.load %filter[%j] : memref<${K}xf32>
    %3 = mulf %0, %1 : f32
    %4 = addf %acc, %3 : f32
    scf.yield %4 : f32
  }
  memref.store %x5, %output[%c5] : memref<${N}xf32>
  //vector.print %x5 : f32
  %x6 = scf.for %j = %c0 to %c3 step %c1 iter_args(%acc = %f0) -> (f32) {
    %start_idx = constant 6 : index
    %idx = addi %start_idx, %j : index
    %0 = memref.load %input[%idx] : memref<${M}xf32>
    %1 = memref.load %filter[%j] : memref<${K}xf32>
    %3 = mulf %0, %1 : f32
    %4 = addf %acc, %3 : f32
    scf.yield %4 : f32
  }
  memref.store %x6, %output[%c6] : memref<${N}xf32>
  //vector.print %x6 : f32
  %x7 = scf.for %j = %c0 to %c3 step %c1 iter_args(%acc = %f0) -> (f32) {
    %start_idx = constant 7 : index
    %idx = addi %start_idx, %j : index
    %0 = memref.load %input[%idx] : memref<${M}xf32>
    %1 = memref.load %filter[%j] : memref<${K}xf32>
    %3 = mulf %0, %1 : f32
    %4 = addf %acc, %3 : f32
    scf.yield %4 : f32
  }
  memref.store %x7, %output[%c7] : memref<${N}xf32>
  //vector.print %x7 : f32
  %x8 = scf.for %j = %c0 to %c3 step %c1 iter_args(%acc = %f0) -> (f32) {
    %start_idx = constant 8 : index
    %idx = addi %start_idx, %j : index
    %0 = memref.load %input[%idx] : memref<${M}xf32>
    %1 = memref.load %filter[%j] : memref<${K}xf32>
    %3 = mulf %0, %1 : f32
    %4 = addf %acc, %3 : f32
    scf.yield %4 : f32
  }
  memref.store %x8, %output[%c8] : memref<${N}xf32>
  //vector.print %x8 : f32
  %x9 = scf.for %j = %c0 to %c3 step %c1 iter_args(%acc = %f0) -> (f32) {
    %start_idx = constant 9 : index
    %idx = addi %start_idx, %j : index
    %0 = memref.load %input[%idx] : memref<${M}xf32>
    %1 = memref.load %filter[%j] : memref<${K}xf32>
    %3 = mulf %0, %1 : f32
    %4 = addf %acc, %3 : f32
    scf.yield %4 : f32
  }
  memref.store %x9, %output[%c9] : memref<${N}xf32>
  //vector.print %x9 : f32
  %x10 = scf.for %j = %c0 to %c3 step %c1 iter_args(%acc = %f0) -> (f32) {
    %start_idx = constant 10 : index
    %idx = addi %start_idx, %j : index
    %0 = memref.load %input[%idx] : memref<${M}xf32>
    %1 = memref.load %filter[%j] : memref<${K}xf32>
    %3 = mulf %0, %1 : f32
    %4 = addf %acc, %3 : f32
    scf.yield %4 : f32
  }
  memref.store %x10, %output[%c10] : memref<${N}xf32>
  //vector.print %x10 : f32
  %x11 = scf.for %j = %c0 to %c3 step %c1 iter_args(%acc = %f0) -> (f32) {
    %start_idx = constant 11 : index
    %idx = addi %start_idx, %j : index
    %0 = memref.load %input[%idx] : memref<${M}xf32>
    %1 = memref.load %filter[%j] : memref<${K}xf32>
    %3 = mulf %0, %1 : f32
    %4 = addf %acc, %3 : f32
    scf.yield %4 : f32
  }
  memref.store %x11, %output[%c11] : memref<${N}xf32>
  //vector.print %x11 : f32
  %x12 = scf.for %j = %c0 to %c3 step %c1 iter_args(%acc = %f0) -> (f32) {
    %start_idx = constant 12 : index
    %idx = addi %start_idx, %j : index
    %0 = memref.load %input[%idx] : memref<${M}xf32>
    %1 = memref.load %filter[%j] : memref<${K}xf32>
    %3 = mulf %0, %1 : f32
    %4 = addf %acc, %3 : f32
    scf.yield %4 : f32
  }
  memref.store %x12, %output[%c12] : memref<${N}xf32>
  //vector.print %x12 : f32
  %x13 = scf.for %j = %c0 to %c3 step %c1 iter_args(%acc = %f0) -> (f32) {
    %start_idx = constant 13 : index
    %idx = addi %start_idx, %j : index
    %0 = memref.load %input[%idx] : memref<${M}xf32>
    %1 = memref.load %filter[%j] : memref<${K}xf32>
    %3 = mulf %0, %1 : f32
    %4 = addf %acc, %3 : f32
    scf.yield %4 : f32
  }
  memref.store %x13, %output[%c13] : memref<${N}xf32>
  //vector.print %x13 : f32
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
  call @conv1d_scalar_unrolled(%input, %filter, %output) : (memref<${M}xf32>, memref<${K}xf32>, memref<${N}xf32>) -> ()

  %t_start = call @rtclock() : () -> f64
  scf.for %arg0 = %c0 to %iters step %c1 {
    call @conv1d_scalar_unrolled(%input, %filter, %output) : (memref<${M}xf32>, memref<${K}xf32>, memref<${N}xf32>) -> ()
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


