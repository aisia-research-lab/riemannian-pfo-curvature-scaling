from jacobi_distortion import self_test, evaluate_case
assert self_test()
q=evaluate_case(1.0,2.0,90)
print('self-test: PASS')
print('kappa=1 r=2 transverse: numerical=',q['directional_num'],' exact=',q['directional_exact'])
