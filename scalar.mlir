func @conv1d_scalar(%input : memref<${M}xf32>, %filter : memref<${K}xf32>, %output : memref<${N}xf32>) 
  attributes { passthrough = ["noinline", ["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c3 = constant ${K} : index
  %c14 = constant ${N} : index
  scf.for %i = %c0 to %c14 step %c1 {
    %y = constant 0.0 : f32
    %x = scf.for %j = %c0 to %c3 step %c1 iter_args(%acc = %y) -> (f32) {
      %idx = addi %i, %j : index
      %0 = memref.load %input[%idx] : memref<${M}xf32>
      %1 = memref.load %filter[%j] : memref<${K}xf32>
      %3 = mulf %0, %1 : f32
      %4 = addf %acc, %3 : f32
      scf.yield %4 : f32
    }
    memref.store %x, %output[%i] : memref<${N}xf32>
  }
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
  scf.for %arg0 = %c0 to %iters step %c1 {
    call @conv1d_scalar(%input, %filter, %output) : (memref<${M}xf32>, memref<${K}xf32>, memref<${N}xf32>) -> ()
  }
  %t_start = call @rtclock() : () -> f64
  scf.for %arg0 = %c0 to %iters step %c1 {
    call @conv1d_scalar(%input, %filter, %output) : (memref<${M}xf32>, memref<${K}xf32>, memref<${N}xf32>) -> ()
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