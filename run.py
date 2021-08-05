#!/usr/bin/env python3
import argparse
import os
import subprocess

option_choices = [
  'scalar', 
  'multi_reduction', 
  'shuffled_contraction_parallel_reduction', 
  'shuffled_contraction_reduction_parallel'
]

cpu_choices = [
  'nehalem',
  'sandybridge',
  'ivybridge',
  'haswell',
  'skylake-avx512',
  'cortex-a34',
  'cortex-a53',
  'cortex-a78'
]

arch_choices = [
  'x86-64',
  'arm64'
]

mlir_opt_flags = [
  '-test-vector-multi-reduction-lowering-patterns',
  '-test-vector-contraction-conversion=vector-outerproduct=1',
  '-canonicalize',
  '-convert-linalg-to-loops',
  '-convert-vector-to-llvm',
  '-convert-memref-to-llvm',
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

opt_flags =  lambda args : [
  '-O3',
  f'-march={args.a}',
  f'-mcpu={args.c}',
]

llc_flags = lambda args : [
  '-O3',
  '--function-sections',# Important to isolate functions and pass to objdump
  '-filetype=obj',
  f'-march={args.a}',
  f'-mcpu={args.c}',
]

# Note: llvm-mca requires a processor to run properly,
# otherwise it will default to the host processor and make a mess.
llvm_mca_flags = lambda args : [
    '--all-stats',
    '--timeline',
    '--bottleneck-analysis',
    f'-march={args.a}',
    f'-mcpu={args.c}',
]

objdump_flags = [
    '-d', 
    '--section=.text.compute',
    '--no-leading-headers',
    '--no-show-raw-insn',
    '--no-leading-addr',
    '-M', 'att',
    ]

def objdump_and_llvm_mca(args, obj_file):

    # Run llvm-objdump to produce the interesting portion of asm.
    asm_file = args.o + '.S'
    f = open(asm_file, 'w')
    objdump = os.path.join(args.m, 'bin/llvm-objdump')
    p = subprocess.Popen([objdump] + objdump_flags + [obj_file], stdout=subprocess.PIPE)
    print(" ".join(p.args))
    p = subprocess.Popen(['tail', '-n', '+7'], stdin=p.stdout, stdout=f)
    p.wait()
    f.close()

    # Run llvm-objdump to produce the full asm for debugging.
    full_asm_file = args.o + '-full.S'
    f = open(full_asm_file, 'w')
    subprocess.run([objdump] + ['-d', obj_file], stdout=f)
    f.close()
    
    # Run llvm-mca on asm
    llvm_mca_out_file = args.o + '_llvm_mca.out'
    llvm_mca = os.path.join(args.m, 'bin/llvm-mca')
    p = subprocess.run([llvm_mca] + llvm_mca_flags(args) + [asm_file] + ['-o'] + [llvm_mca_out_file])
    print(" ".join(p.args))
    
    # Dump 10 lines of llvm-mca.
    subprocess.run(['head', '-n', '10', llvm_mca_out_file])

def is_valid_fn(fn, mlir_file):
    with open(mlir_file, 'r') as f:
        data = f.read()
    return 'compute_v' + fn in data

def compile_to_llvm_dialect(args):

    mlir_file = os.path.basename(args.o + '.mlir')
    # Check if fn is valid, if specified
    if args.f is not None:
        if not is_valid_fn(args.f, mlir_file):
            raise ValueError(f"compute_v{args.f} is not defined in {mlir_file}")
    else:
        # Set default option to 'compute_v1'
        args.f = 1

    if 'scalar' not in args.o:
        print(f"Running compute_v{args.f} in {mlir_file}")

    # Run mlir-opt.
    mlir_opt = os.path.join(args.m, 'bin/mlir-opt')
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
    psed = subprocess.Popen(['sed'] + ['s/${TARGET_CPU}/, ["target-cpu", "' + args.c + '"]/g'] , stdin=psed.stdout, stdout=subprocess.PIPE)
    psed = subprocess.Popen(['sed'] + ['s/${PREFER_VECTOR_WIDTH}/, ["prefer-vector-width", "' + args.v + '"]/g'] , stdin=psed.stdout, stdout=subprocess.PIPE)
    psed = subprocess.Popen(['sed'] + ['s/${N}/16/g'] , stdin=psed.stdout, stdout=subprocess.PIPE)
    psed = subprocess.Popen(['sed'] + ['s/${K}/3/g'] , stdin=psed.stdout, stdout=subprocess.PIPE)
    psed = subprocess.Popen(['sed'] + ['s/${FN}/' + f'{args.f}/g'] , stdin=psed.stdout, stdout=subprocess.PIPE)

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
    prun = subprocess.Popen([opt] + opt_flags(args), stdin=prun.stdout, stdout=subprocess.PIPE)
    print(" ".join(prun.args))

    obj_file = args.o + '.o'
    f = open(obj_file, 'w')
    llc = os.path.join(args.m, 'bin/llc')
    prun = subprocess.Popen([llc] + llc_flags(args) + ['-o'] + [obj_file], stdin=prun.stdout, stdout=f)
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
    if args.a == 'x86-64' :
      run_and_check(args, mlir_file, mlir_outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Utility to evaluate conv1d vectorization options')
    parser.add_argument('-m', '-mlir_build_dir', help='path to mlir build dir', required=True)
    parser.add_argument('-o', '-option', default='scalar', choices=option_choices,
            help='which conv1d vectorization strategy to evaluate')
    parser.add_argument('-v', '-vector_width', default='512', 
            help='preferred vector vector_width to inject in the MLIR examples')
    parser.add_argument('-c', '-cpu', default='skylake-avx512', choices=cpu_choices,
            help='cpu to compile for')
    parser.add_argument('-a', '-arch', default='x86-64', choices=arch_choices,
            help='arch to compile for')
    parser.add_argument('-f', '-fn', help='specifies which compute_v* function to run')
    args = parser.parse_args()
    args.o = 'outputs/' + args.o + '/' + args.o
    run(args)
