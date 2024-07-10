"""
Different ways to run the unit tests with pytest:

i) Run all test (-s prints output)
python3 -m

ii) Run a specific script
python3 -m pytest tests/test_dgp_sparse.py

iii) Run a specific function/class in that script
python3 -m pytest tests/test_dgp_sparse.py -s -k 'TestDGPSparseYX'
python3 -m pytest tests/test_dgp_sparse.py::TestDGPSparseYX

iv) Run a specific method for a test class
python3 -m pytest tests/test_dgp_sparse.py::TestDGPSparseYX::test_gen_yX_output
"""

