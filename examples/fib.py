import os.path
import sys
sys.path.append(os.path.join(os.path.abspath('..')))

import pywebassembly as wasm
import timeit

file_ = open('fibonacci.wasm', 'rb')
bytes_ = memoryview(file_.read())            #can also use bytearray or bytes instead of memoryview
module = wasm.decode_module(bytes_)            #get module as abstract syntax
store = wasm.init_store()                #do this once for each VM instance
externvalstar = []                    #imports, none for fibonacci.wasm
store,moduleinst,ret = wasm.instantiate_module(store,module,externvalstar)
externval = wasm.get_export(moduleinst, "fib")        #we want to call the function "fib"
funcaddr = externval[1]                    #the address of the funcinst for "fib"
args = [["i32.const", 9]]              #list of arguments, one arg in our case
start_time = timeit.default_timer()
store,ret = wasm.invoke_func(store,funcaddr,args)    #finally, invoke the function
print(timeit.default_timer() - start_time)
print(ret)

