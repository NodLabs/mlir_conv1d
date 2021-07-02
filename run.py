#!/usr/bin/env python3
import argparse
import os
import subprocess

objdump_binary = 'objdump'
option_choices = ['scalar', 'scalar_unrolled', 'multi_reduction', 
                  'unrolled_contraction', 'shuffled_contraction_parallel_reduction', 'shuffled_contraction_reduction_parallel']
mlir_opt_flags = [
  '-test-vector-multi-reduction-lowering-patterns',
  '-test-vector-contraction-conversion=vector-outerproduct=1',
  '-canonicalize',
  '-convert-linalg-to-loops',
  '-convert-vector-to-llvm',
  '-convert-scf-to-std',
  '-convert-std-to-llvm',
  '-canonicalize',
]
mlir_cpu_runner_dump_object_flags = lambda build_dir, object_filename : [
  '-O3',
  '-entry-point-result=void',
  f'-shared-libs={build_dir}/lib/libmlir_c_runner_utils.so,{build_dir}/lib/libmlir_runner_utils.so',
  '-dump-object-file',
  f'-object-filename={object_filename}',
]
mlir_cpu_runner_run_flags = lambda build_dir, object_filename : [
  '-O3',
  '-e=main',
  '-entry-point-result=void',
  f'-shared-libs={build_dir}/lib/libmlir_c_runner_utils.so,{build_dir}/lib/libmlir_runner_utils.so',
]
objdump_flags = ['-D']
llvm_mca_flags = []

def profile(args, obj_name):
    # Run objdump to get asm
    dumpfile = args.o + '.dump'
    f = open(dumpfile, 'w')
    subprocess.run([objdump_binary] + objdump_flags + [obj_name], stdout=f)
    f.close()

    # Extract asm of relevant section
    with open(dumpfile, 'r') as f:
        data = f.readlines()

    captured = []
    capturing = False
    for line in data:
        if '<conv1d_' in line:
            capturing = True
        if capturing:
            captured.append(line)
        if capturing and 'retq' in line:
            break

    asm = []
    for line in captured:
        splits = line.split('\t')
        if len(splits) == 3:
            asm.append(splits[-1])
    asm_file = args.o + '.S'
    with open(asm_file, 'w') as f:
        for line in asm:
            f.write(line)

    # Run llvm-mca on asm
    llvm_mca_out_file = args.o + '_llvm_mca.out'
    f = open(llvm_mca_out_file, 'w')
    res = subprocess.run([args.llvm_mca] + llvm_mca_flags + [asm_file], stdout=f)
    f.close()

    with open(llvm_mca_out_file, 'r') as f:
        count = 0
        for line in f.readlines():
            if count < 10:
                print(line.strip())
            count += 1

def compile_and_run(args):
    # Run mlir-opt
    mlir_opt = os.path.join(args.m, 'bin/mlir-opt')
    mlir_file = args.o + '.mlir'
    mlir_outfile = args.o + 'mlir.out'

    print(" ".join(['mkdir'] + ['-p'] + [os.path.dirname(mlir_outfile)]))
    p = subprocess.Popen(['mkdir'] + ['-p'] + [os.path.dirname(mlir_outfile)])
    p.wait()

    f = open(mlir_outfile, 'w')
    cat = subprocess.Popen(['cat'] + [os.path.basename(mlir_file)] , stdout=subprocess.PIPE)
    sed = subprocess.Popen(['sed'] + ['s/${ITERS}/1000000/g'] , stdin=cat.stdout, stdout=subprocess.PIPE)
    sed = subprocess.Popen(['sed'] + ['s/${M}/16/g'] , stdin=sed.stdout, stdout=subprocess.PIPE)
    sed = subprocess.Popen(['sed'] + ['s/${N}/14/g'] , stdin=sed.stdout, stdout=subprocess.PIPE)
    sed = subprocess.Popen(['sed'] + ['s/${K}/3/g'] , stdin=sed.stdout, stdout=subprocess.PIPE)
    subprocess.run([mlir_opt] + mlir_opt_flags + ['-'], stdin=sed.stdout, stdout=f)
    cat.stdout.close()
    f.close()

    # Run mlir-cpu-runner
    mlir_cpu_runner = os.path.join(args.m, 'bin/mlir-cpu-runner')
    obj_name = args.o + '.o'
    subprocess.run([mlir_cpu_runner] + mlir_cpu_runner_run_flags(args.m, obj_name) + [mlir_outfile])
    print(" ".join([mlir_cpu_runner] + mlir_cpu_runner_dump_object_flags(args.m, obj_name) + [mlir_outfile]))
    subprocess.run([mlir_cpu_runner] + mlir_cpu_runner_dump_object_flags(args.m, obj_name) + [mlir_outfile])
    return obj_name

def run(args):
    print(f"Evaluating ... {args.o}.mlir")
    obj_name = compile_and_run(args)
    profile(args, obj_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Utility to evaluate conv1d vectorization options')
    parser.add_argument('-m', '-mlir_build_dir', help='path to mlir build dir', required=True)
    parser.add_argument('-o', '-option', default='scalar', choices=option_choices,
            help='which conv1d vectorization strategy to evaluate')
    parser.add_argument('-llvm_mca', default='llvm-mca', help='llvm-mca binary to use for profiling')
    args = parser.parse_args()
    args.o = 'outputs/' + args.o + '/' + args.o
    run(args)

