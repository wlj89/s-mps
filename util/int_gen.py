"""
    FCIDUMP generation script
	lmfao
"""
import psi4
 
psi4.core.clean() # Clean local scratch files
psi4.set_memory('8000 MB')

dump = 'LiH-cc-pvdz.FCIDUMP'
outfile = 'LiH-cc-pvdz.psi4out'
psi4.core.set_output_file(outfile)
# There are many ways to specify geometry, see documentation


h2_geom = """
H 0 0 0 
H 0 0 0.74
"""

HF_geom = """
H 0 0 0 
F 0 0 0.917
"""

h2o_geom = """
O
H 1 0.96
H 1 0.96 2 104.5
"""
Li2_geom = """
Li 0 0 0 
Li 0 0 2.67
"""

LiH_geom = """
H 0 0 0 
Li 0 0 2.01
"""

mol = psi4.geometry(LiH_geom)

# Default SCF_TYPE is density fitting, which produces slightly different results than other packages like PySCF.
# Provide 'SCF_TYPE':'DIRECT' in the option dictionary if it's concerning
psi4.set_options({'basis':'cc-pvdz','SCF_TYPE':'DIRECT'}) 
E, wfn = psi4.energy('scf', return_wfn=True)

# oe_ints has to be specified exactly like this
psi4.fcidump(wfn, fname=dump, oe_ints=['EIGENVALUES'])

E_fci = psi4.energy('fci')

print(E)
print(E_fci)
