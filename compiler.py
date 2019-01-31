from llvmlite import ir
import llvmlite.binding as llvm
from ctypes import CFUNCTYPE, c_int32

# based on
# https://ian-bertolacci.github.io/llvm/llvmlite/python/compilers/programming/2016/03/06/LLVMLite_fibonacci.html
# http://dev.stephendiehl.com/numpile/
# http://llvmlite.pydata.org/en/latest/user-guide/ir/ir-builder.html

INSTRUCTIONS = {
  'block': 'visit_Block',
  'end': 'visit_End',
  'loop': 'visit_Loop',
  'br_if': 'visit_BranchIf',
  'get_local': 'visit_GetLocal',
  'set_local': 'visit_SetLocal',
  'tee_local': 'visit_TeeLocal',
  'i32.const': 'visit_Const',
  'i32.lt_s': 'visit_Prim',
  'i32.add': 'visit_Prim'
}

class LLVMEmitter(object):
  def __init__(self, argtypes):
    self.module = None
    self.function = None             # LLVM Function
    self.builder = None
    # self.builder.block is the current block
    self.locals = {}                 # Local variables
    self.stack_vars = []
    self.control_stack = []

    #self.retty = retty               # Return type
    # TODO: return types
    self.argtypes = argtypes             # Argument types

  def start_function(self, body):
    # Create a 32bit wide int type
    int_type = ir.IntType(32);
    # Create a int -> int function
    fn_int_to_int_type = ir.FunctionType( int_type, [int_type] )

    self.module = ir.Module(name="m_fibonacci_example")
    self.function = ir.Function(self.module, fn_int_to_int_type, name="fn_fib")
    self.builder = ir.IRBuilder()
    entry_block = self.function.append_basic_block( name="entry" )
    self.control_stack.append(entry_block)
    self.builder.position_at_end(entry_block)

    print("calling visit on body: {}".format(body))
    for instr in body:
      self.visit(instr)
    return self.module

  def visit_Loop(self, node):
    loop_block = self.function.append_basic_block('loop.init')

    self.builder.branch(loop_block)

    self.control_stack.append([loop_block, 'loop'])
    print("got loop init block: {}".format(loop_block))
    self.builder.position_at_start(loop_block)
    
    block_instrs = node[2]
    print("iterating over loop block_instrs...")
    for instr in block_instrs:
      self.visit(instr)

  def visit_GetLocal(self, node):
    local_index = node[1] # 0-based index
    if local_index + 1 <= len(self.argtypes):
      # getting an input param
      # wasm does tee_local on input params
      # so we need to initialize an llvm var and copy the input param there
      if local_index not in self.locals:
        stackint = self.builder.alloca(ir.IntType(32))
        self.locals[local_index] = stackint
        self.builder.store(self.function.args[local_index], stackint)
        myint = self.builder.load(self.locals[local_index])
        self.stack_vars.append(myint)
      else:
        myint = self.builder.load(self.locals[local_index])
        self.stack_vars.append(myint)
    else:
      # getting a local var
      # local var should already be initialized in self.locals
      # self.locals[local_index] should be an allocated stackint
      myint = self.builder.load(self.locals[local_index])
      self.stack_vars.append(myint)


  def visit_SetLocal(self, node):
    local_index = node[1]
    if local_index not in self.locals:
      stackint = self.builder.alloca(ir.IntType(32))
      self.locals[local_index] = stackint
      # self.stack_vars[-1] is either an ir.constant or ir.argument
      self.builder.store(self.stack_vars[-1], stackint)
    else:
      # self.locals[local_index] is already an allocated stackint
      stackint = self.locals[local_index]
      self.builder.store(self.stack_vars[-1], stackint)
    # TODO: check that self.stack_vars[-1] is the expected type
    self.stack_vars.pop()
 
  def visit_TeeLocal(self, node):
    local_index = node[1]
    stackint = self.locals[local_index]
    self.builder.store(self.stack_vars[-1], stackint)

  def visit_Const(self, node):
    # node looks like ['i32.const', 4294967295]
    const_var = ir.Constant(ir.IntType(32), node[1])
    self.stack_vars.append(const_var)

  def visit_Block(self, node):
    # node looks like [['block'], [], [...instructions...]]
    block_name = "block{}".format(len(self.control_stack))
    print("calling append_basic_block...")
    new_block = self.function.append_basic_block(name=block_name)
    self.control_stack.append([new_block, 'block'])
    
    #self.builder.position_at_end(self.builder.block)

    #TODO: need to append the "end_block" here
    # branching to a wasm block label goes to the end of the block..
    
    self.builder.branch(new_block)
    self.builder.position_at_start(new_block)

    block_instrs = node[2]
    print("iterating over block_instrs...")
    for instr in block_instrs:
      self.visit(instr)

  def visit_BranchIf(self, node):
    branch_label = node[1]
    # branch_label is an index corresponding to control_stack
    # if branch label is a wasm loop, then go to beginning of the loop
    # if branch label is a wasm block, then go to end of the block
    conditional_val = self.stack_vars[-1]
    self.stack_vars.pop()
    control_stack_index = (-1 * branch_label) - 1
    # if label is 0, then we want the current block, control_stack[-1]
    # if branch label is 1, then want one level out, control_stack[-2]
    target_block = self.control_stack[control_stack_index]
    # target_block looks like [basic_block, "block" or "loop"]

    # do we need a Phi node to solve this properly?
    # http://llvmlite.pydata.org/en/latest/user-guide/ir/ir-builder.html#phi

    if target_block[1] == "block":
      # we want to jump to the end of the target block
      # i.e. branch to the block that follows the target block
      # but that block hasn't been created yet, so we can't reference it?
      a = "1"
      # if pred is true, branch to target_block. else branch to current block?
      # self.builder.cbranch(pred, target_block, self.builder.block)
      # TODO: finish this
    elif target_block[1] == "loop":
      # we want to jump to the beginning of the loop
      # wasm only permits jumping to a loop that you are already in
      # so we know the loop block has already been created
      #
      # trunc on the conditional var is broken.
      # predicate = self.builder.trunc(conditional_val, ir.IntType(1))
      # we should do icmp or something instead
      predicate = self.builder.icmp_signed(cmpop=">", lhs=conditional_val, rhs=ir.Constant(ir.IntType(32), 0))
      with self.builder.if_then(predicate):
        # now we are at end of the newly created conditional block
        # TODO: should we jump to the loop "entry" block?
        self.builder.branch(target_block[0])
      # now we are start of continuation block
    # end of br_if
  
  def visit_End(self, node):
    # jump to continuation block of the outer block?
    # basic blocks must have a terminator.
    #print("visit_End self.control_stack: {}".format(self.control_stack))
    if len(self.control_stack) > 1:
      self.control_stack.pop()
      control_index = len(self.control_stack)
      label = "contin-{}".format(control_index)
      # TODO: use unique names for continuation blocks
      continue_block = self.function.append_basic_block(label)
      self.builder.branch(continue_block)
      self.builder.position_at_start(continue_block)
    else:
      # print("end of function.. control_stack should have one element: {}".format(self.control_stack))
      self.control_stack.pop()
      # TODO: handle cases where function doesnt return anything
      # self.builder.ret_void()
      #print("there should be a val left on the stack: {}".format(self.stack_vars))
      ret_val = self.stack_vars[-1]
      self.builder.ret(ret_val)
      self.stack_vars.pop()


  def visit_Prim(self, node):
    node_fn = node[0]
    if node_fn == "i32.lt_s":
      lhs = self.stack_vars[-1]
      rhs = self.stack_vars[-2]
      # TODO: check that lhs and rhs are correct stack slots
      lt_s_result = self.builder.icmp_signed(cmpop="<", lhs=lhs, rhs=rhs)
      self.stack_vars.pop()
      self.stack_vars.pop()
      self.stack_vars.append(lt_s_result)
    elif node_fn == "mult#":
      a = self.visit(node.args[0])
      b = self.visit(node.args[1])
      if a.type == double_type:
        return self.builder.fmul(a, b)
      else:
        return self.builder.mul(a, b)
    elif node_fn == "i32.add":
      a = self.stack_vars[-1]
      b = self.stack_vars[-2]
      #print("calling self.builder.add...")
      #print("a: {}".format(a))
      #print("b: {}".format(b))
      add_result = self.builder.add(a, b)
      self.stack_vars.pop()
      self.stack_vars.pop()
      self.stack_vars.append(add_result)
    else:
      raise NotImplementedError

  def visit(self, node):
    #print("visiting node: {}".format(node))
    node_instr = node[0]
    if node_instr in INSTRUCTIONS:
      handler = INSTRUCTIONS[node_instr]
      #print("calling visit handler: {}  on node {}".format(handler, node))
      return getattr(self, handler)(node)
    else:
      raise NotImplementedError


def executeMachineCode(module, operands):
  """
  Execute generated code.
  """
  print("executing llvm code: {}".format(module))
  # initialize the LLVM machine
  # These are all required (apparently)
  llvm.initialize()
  llvm.initialize_native_target()
  llvm.initialize_native_asmprinter()

  # Create engine and attach the generated module
  # Create a target machine representing the host
  target = llvm.Target.from_default_triple()
  target_machine = target.create_target_machine()
  # And an execution engine with an empty backing module
  backing_mod = llvm.parse_assembly("")
  engine = llvm.create_mcjit_compiler(backing_mod, target_machine)

  # Parse our generated module
  mod = llvm.parse_assembly( str( module ) )
  print("verifying llvm module...")
  mod.verify()
  print("module verified!")
  # Now add the module and make sure it is ready for execution
  engine.add_module(mod)
  engine.finalize_object()

  # Look up the function pointer (a Python int)
  func_ptr = engine.get_function_address("fn_fib")

  # Run the function via ctypes
  c_fn_fib = CFUNCTYPE(c_int32, c_int32)(func_ptr)
  
  n = operands[0]
  result = c_fn_fib(n)
  print( "c_fn_fib({0}) = {1}".format(n, result) )
  return result


def compileAndRunWasmFunc(config):
  # takes same inputs as spec_invoke_function_address()
  S = config["S"]
  F = config["F"]
  instrstar = config["instrstar"]
  idx = config["idx"]
  operand_stack  = config["operand_stack"]
  control_stack  = config["control_stack"]
  a=instrstar[idx][1]
  f = S["funcs"][a]
  t1n, t2m = f["type"]  # [['i32'], ['i32']]
  tstar = f["code"]["locals"]
  instrstarend = f["code"]["body"]
  input_params = f["type"][0] # ['i32']
  return_types = f["type"][1]
  # f looks like {'type': [['i32'], ['i32']], 'module': {'types': [[['i32'], ['i32']]], 'funcaddrs': [0], 'tableaddrs': [0], 'memaddrs': [0], 'globaladdrs': [], 'exports': [{'name': 'memory', 'value': ['mem', 0]}, {'name': 'fib', 'value': ['func', 0]}]}, 'code': {'type': 0, 'locals': ['i32', 'i32', 'i32'], 'body': [['i32.const', 1], ['set_local', 1], ['block', [], [['get_local', 0], ['i32.const', 1], ['i32.lt_s'], ['br_if', 0], ['i32.const', 1], ['set_local', 3], ['i32.const', 0], ['set_local', 2], ['loop', [], [['get_local', 2], ['get_local', 3], ['i32.add'], ['set_local', 1], ['get_local', 3], ['set_local', 2], ['get_local', 1], ['set_local', 3], ['get_local', 0], ['i32.const', 4294967295], ['i32.add'], ['tee_local', 0], ['br_if', 0], ['end']]], ['end']]], ['get_local', 1], ['end']]}}

  codegen = LLVMEmitter(input_params)
  #print("calling start_function...")
  codegen.start_function(f["code"]["body"])
  llvm_mod = codegen.module
  result = executeMachineCode(llvm_mod, operand_stack)
  return result
