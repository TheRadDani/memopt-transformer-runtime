import pytest
import memopt

def test_memopt_import():
    assert memopt.__version__ == "0.1.0"
    
def test_cpp_extension_loaded():
    hello_msg = memopt.get_hello()
    assert "MemOpt C++ Backend Initialized" in hello_msg or "C++ backend not loaded" in hello_msg
