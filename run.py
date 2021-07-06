#!/usr/bin/env python3
import argparse
import os
import subprocess

option_choices = ['scalar', 'multi_reduction', 
                  'shuffled_contraction_parallel_reduction', 'shuffled_contraction_reduction_parallel']

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

mlir_cpu_runner_run_flags = lambda build_dir : [
  '-O3',
  '-e=main',
  '-entry-point-result=void',
  f'-shared-libs={build_dir}/lib/libmlir_c_runner_utils.so,{build_dir}/lib/libmlir_runner_utils.so',
]

opt_flags_skylake_avx512 = [
  '-O3',
  '-march=x64-64',
  '-mcpu=skylake-avx512',
]
opt_flags_armv8 = [
  '-O3',
  '-march=arm64',
]
opt_flags = opt_flags_armv8

llc_flags_common = [
  '-O3',
  '--function-sections',# Important to isolate functions and pass to objdump
  '-filetype=obj'
]
llc_flags_skylake_avx512 = [
  # For some reason, x86-64 does not work ...
  #'-march=x64-64',
  '-mcpu=skylake-avx512',
]
llc_flags_armv8 = [
  '-march=arm64',
]
llc_flags = llc_flags_common + llc_flags_armv8


# Note: llvm-mca requires a processor to run properly,
# otherwise it will default to the host processor and make a mess.
llvm_mca_common = [
    '--all-stats',
    '--timeline',
    '--bottleneck-analysis'
  ]
llvm_mca_flags_skylake_avx512 = [
  # For some reason, x86-64 does not work ...
  #'-march=x64-64',
  '-mcpu=skylake-avx512',
]
llvm_mca_flags_armv8 = [
  '-march=arm64',
  '-mcpu=cortex-a34',
]
llvm_mca_flags = llvm_mca_common + llvm_mca_flags_armv8

objdump_flags = [
    '-d', 
    '--section=.text.compute',
    '--no-leading-headers',
    '--no-show-raw-insn',
    '--no-leading-addr',
    '-M', 'att',
    ]

def objdump_and_llvm_mca(args, obj_file):

    # Run llvm-objdump to produce asm.
    asm_file = args.o + '.S'
    f = open(asm_file, 'w')
    objdump = os.path.join(args.m, 'bin/llvm-objdump')
    p = subprocess.Popen([objdump] + objdump_flags + [obj_file], stdout=subprocess.PIPE)
    print(" ".join(p.args))
    p = subprocess.Popen(['tail', '-n', '+7'], stdin=p.stdout, stdout=f)
    p.wait()
    f.close()

    # Run llvm-mca on asm
    llvm_mca_out_file = args.o + '_llvm_mca.out'
    llvm_mca = os.path.join(args.m, 'bin/llvm-mca')
    p = subprocess.run([llvm_mca] + llvm_mca_flags + [asm_file] + ['-o'] + [llvm_mca_out_file])
    print(" ".join(p.args))
    
    # Dump 10 lines of llvm-mca.
    subprocess.run(['head', '-n', '10', llvm_mca_out_file])


def compile_to_llvm_dialect(args):
    # Run mlir-opt.
    mlir_opt = os.path.join(args.m, 'bin/mlir-opt')
    mlir_file = os.path.basename(args.o + '.mlir')
    mlir_outfile = args.o + '-mlir.out'

    p = subprocess.Popen(['mkdir'] + ['-p'] + [os.path.dirname(mlir_outfile)])
    print(" ".join(p.args))
    p.wait()

    # For now values are hardcoded because tests are written with size knowledge.
    # In the future we may want to expand this behavior.
    psed = subprocess.Popen(['cat'] + [mlir_file] , stdout=subprocess.PIPE)
    psed = subprocess.Popen(['sed'] + ['s/${ITERS}/1000000/g'] , stdin=psed.stdout, stdout=subprocess.PIPE)
    psed = subprocess.Popen(['sed'] + ['s/${M}/18/g'] , stdin=psed.stdout, stdout=subprocess.PIPE)
    psed = subprocess.Popen(['sed'] + ['s/${N}/16/g'] , stdin=psed.stdout, stdout=subprocess.PIPE)
    psed = subprocess.Popen(['sed'] + ['s/${K}/3/g'] , stdin=psed.stdout, stdout=subprocess.PIPE)

    p = subprocess.run([mlir_opt] + mlir_opt_flags + ['-o'] + [mlir_outfile], stdin=psed.stdout)
    print(" ".join(p.args))

    return mlir_file, mlir_outfile

# Run opt and llc to produce a .o
def compile_to_object(args, mlir_outfile):
    mlir_translate = os.path.join(args.m, 'bin/mlir-translate')
    ll_file = args.o + '.ll'
    prun = subprocess.run([mlir_translate] + ['--mlir-to-llvmir'] + [mlir_outfile] + ['-o'] + [ll_file])
    print(" ".join(prun.args))

    opt = os.path.join(args.m, 'bin/opt')
    prun = subprocess.Popen(['cat'] + [ll_file], stdout=subprocess.PIPE)
    prun = subprocess.Popen([opt] + opt_flags, stdin=prun.stdout, stdout=subprocess.PIPE)
    print(" ".join(prun.args))

    obj_file = args.o + '.o'
    f = open(obj_file, 'w')
    llc = os.path.join(args.m, 'bin/llc')
    prun = subprocess.Popen([llc] + llc_flags + ['-o'] + [obj_file], stdin=prun.stdout, stdout=f)
    print(" ".join(prun.args))
    prun.wait()
    f.close()

    return obj_file

# Run mlir-cpu-runner and check results.
def run_and_check(args, mlir_file, mlir_outfile):
    mlir_cpu_runner = os.path.join(args.m, 'bin/mlir-cpu-runner')
    filecheck = os.path.join(args.m, 'bin/FileCheck')
    print("########## Run MLIR compute function ##########")
    import tempfile
    with tempfile.NamedTemporaryFile() as temp:
        prun = subprocess.Popen([mlir_cpu_runner] + mlir_cpu_runner_run_flags(args.m) + [mlir_outfile], stdout=temp)
        prun.wait()
        prun = subprocess.Popen(['cat', str(temp.name)], stdout=subprocess.PIPE)
        prun = subprocess.Popen([filecheck] + [mlir_file], stdin=prun.stdout)
        prun.wait()
        if prun.returncode != 0:
            print("Result did not check")
            exit(1)
        subprocess.run(['cat', str(temp.name)])
    print("########## DONE Run MLIR compute function ##########")

def run(args):
    print(f"Evaluating ... {args.o}.mlir")

    # Preprocess and compile with mlir-opt.
    mlir_file, mlir_outfile = compile_to_llvm_dialect(args)
    obj_file = compile_to_object(args, mlir_outfile)
    objdump_and_llvm_mca(args, obj_file)
    # run_and_check(args, mlir_file, mlir_outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Utility to evaluate conv1d vectorization options')
    parser.add_argument('-m', '-mlir_build_dir', help='path to mlir build dir', required=True)
    parser.add_argument('-o', '-option', default='scalar', choices=option_choices,
            help='which conv1d vectorization strategy to evaluate')
    parser.add_argument('-llvm_mca', default='llvm-mca', help='llvm-mca binary to use for profiling')
    args = parser.parse_args()
    args.o = 'outputs/' + args.o + '/' + args.o
    run(args)