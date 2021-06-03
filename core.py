import random
import numpy
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

# TODO: move to kernel.
(V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, VA, VB, VC, VD, VE, VF) = range(16)
(I, PC, SP, DT) = range(4)

# TODO: move to kernel.
DISPLAY_WIDTH = 64
DISPLAY_HEIGHT = 32

EXIT_TERMINATION = 0
EXIT_REDRAW = 1

class Core:
  GEN_REG_COUNT = 16
  CPU_REG_COUNT = 4

  HP_FLAGS_SIZE = 16
  STACK_SIZE = 16
  MEMORY_SIZE = 4096

  def __init__(self,
               cuda_device: int = 0,
               stop_on_zero: bool = True,
               sub_vt_compat: bool = True,
               stop_on_infinite_loop: bool = True,
               pc_start: int = 0x0200):
    self._cuda_device = cuda_device
    self._stop_on_zero = stop_on_zero
    self._sub_vt_compat = sub_vt_compat
    self._stop_on_infinite_loop = stop_on_infinite_loop
    self._pc_start = pc_start

    self.reset()


  def reset(self):
    self._config = numpy.array([self._stop_on_zero, self._sub_vt_compat, self._stop_on_infinite_loop], dtype=numpy.bool8)
    self._rng = create_xoroshiro128p_states(1, seed=int(random.random() * 1000))
    self._gen_reg = numpy.zeros(Core.GEN_REG_COUNT, dtype=numpy.uint8)
    self._cpu_reg = numpy.zeros(Core.CPU_REG_COUNT, dtype=numpy.uint16)
    self._hp_flags = numpy.zeros(Core.HP_FLAGS_SIZE, dtype=numpy.uint8)
    self._stack = numpy.zeros(Core.STACK_SIZE, dtype=numpy.uint8)
    self._memory = numpy.zeros(Core.MEMORY_SIZE, dtype=numpy.uint8)
    self._display = numpy.zeros(DISPLAY_WIDTH * DISPLAY_HEIGHT, dtype=numpy.uint8)
    self._output = numpy.zeros(1, dtype=numpy.uint8)

    self._cpu_reg[PC] = self._pc_start

    fonts = [
      [0xF0, 0x90, 0x90, 0x90, 0xF0], # 0
      [0x20, 0x60, 0x20, 0x20, 0x70], # 1
      [0xF0, 0x10, 0xF0, 0x80, 0xF0], # 2
      [0xF0, 0x10, 0xF0, 0x10, 0xF0], # 3
      [0x90, 0x90, 0xF0, 0x10, 0x10], # 4
      [0xF0, 0x80, 0xF0, 0x10, 0xF0], # 5
      [0xF0, 0x80, 0xF0, 0x90, 0xF0], # 6
      [0xF0, 0x10, 0x20, 0x40, 0x40], # 7
      [0xF0, 0x90, 0xF0, 0x90, 0xF0], # 8
      [0xF0, 0x90, 0xF0, 0x10, 0xF0], # 9
      [0xF0, 0x90, 0xF0, 0x90, 0x90], # A
      [0xE0, 0x90, 0xE0, 0x90, 0xE0], # B
      [0xF0, 0x80, 0x80, 0x80, 0xF0], # C
      [0xE0, 0x90, 0x90, 0x90, 0xE0], # D
      [0xF0, 0x80, 0xF0, 0x80, 0xF0], # E
      [0xF0, 0x80, 0xF0, 0x80, 0x80], # F
    ]

    for index in range(len(fonts)):
      self._memory[index*5:(index+1)*5] = numpy.array(fonts[index])


  def load(self, program):
    position = self._pc_start
    for instruction in program:
      self._memory[position] = instruction
      position += 1


  def run(self, redraw_func = None):
    cuda.select_device(self._cuda_device)

    # Copy everything to the gpu
    config_gpu = cuda.to_device(self._config)
    rng_gpu = cuda.to_device(self._rng)
    cpu_reg_gpu = cuda.to_device(self._cpu_reg)
    gen_reg_gpu = cuda.to_device(self._gen_reg)
    hp_flags_gpu = cuda.to_device(self._hp_flags)
    stack_gpu = cuda.to_device(self._stack)
    memory_gpu = cuda.to_device(self._memory)
    display_gpu = cuda.to_device(self._display)
    output_gpu = cuda.to_device(self._output)

    while True:
      # Call the program and get the exit result
      Core.kernel[1, 1](config_gpu, rng_gpu, cpu_reg_gpu, gen_reg_gpu, hp_flags_gpu, stack_gpu, memory_gpu, display_gpu, output_gpu)
      output_gpu.copy_to_host(self._output)

      # Check the reason why we exited the kernel.
      if self._output[0] == EXIT_TERMINATION:
        # We want to leave the program.
        break
      elif self._output[0] == EXIT_REDRAW and redraw_func is not None:
        # We want to force a redraw. Copy the data back and update the image.
        display_gpu.copy_to_host(self._display)
        redraw_func(self._display)

    # Copy the state back to the the OS
    cpu_reg_gpu.copy_to_host(self._cpu_reg)
    gen_reg_gpu.copy_to_host(self._gen_reg)
    hp_flags_gpu.copy_to_host(self._hp_flags)
    stack_gpu.copy_to_host(self._stack)
    memory_gpu.copy_to_host(self._memory)
    display_gpu.copy_to_host(self._display)


  def dump(self):
    print("General registers:")
    print(f"  V0 = {hex(self._gen_reg[V0])} ({self._gen_reg[V0]})")
    print(f"  V1 = {hex(self._gen_reg[V1])} ({self._gen_reg[V1]})")
    print(f"  V2 = {hex(self._gen_reg[V2])} ({self._gen_reg[V2]})")
    print(f"  V3 = {hex(self._gen_reg[V3])} ({self._gen_reg[V3]})")
    print(f"  V4 = {hex(self._gen_reg[V4])} ({self._gen_reg[V4]})")
    print(f"  V5 = {hex(self._gen_reg[V5])} ({self._gen_reg[V5]})")
    print(f"  V6 = {hex(self._gen_reg[V6])} ({self._gen_reg[V6]})")
    print(f"  V7 = {hex(self._gen_reg[V7])} ({self._gen_reg[V7]})")
    print(f"  V8 = {hex(self._gen_reg[V8])} ({self._gen_reg[V8]})")
    print(f"  V9 = {hex(self._gen_reg[V9])} ({self._gen_reg[V9]})")
    print(f"  VA = {hex(self._gen_reg[VA])} ({self._gen_reg[VA]})")
    print(f"  VB = {hex(self._gen_reg[VB])} ({self._gen_reg[VB]})")
    print(f"  VC = {hex(self._gen_reg[VC])} ({self._gen_reg[VC]})")
    print(f"  VD = {hex(self._gen_reg[VD])} ({self._gen_reg[VD]})")
    print(f"  VE = {hex(self._gen_reg[VE])} ({self._gen_reg[VE]})")
    print(f"  VF = {hex(self._gen_reg[VF])} ({self._gen_reg[VF]})")

    print("CPU registers:")
    print(f"  I  = {hex(self._cpu_reg[I])} ({self._cpu_reg[I]})")
    print(f"  PC = {hex(self._cpu_reg[PC])} ({self._cpu_reg[PC]})")
    print(f"  SP = {hex(self._cpu_reg[SP])} ({self._cpu_reg[SP]})")

    print("HP flags:", self._hp_flags)

    print("Stack:", self._stack)

    print("Memory:", self._memory)

    print("Display:", self._display)


  @staticmethod
  @cuda.jit
  def kernel(config, rng, cpu_registers, gen_registers, hp_flags, stack, memory, display, exit_reason):
    (stop_on_zero, sub_vf_compat, stop_on_infinite_loop) = config

    while True:
      instruction = ((memory[cpu_registers[PC]] << 8) & 0xFF00) + memory[cpu_registers[PC] + 1]
      #print("Executing PC", cpu_registers[PC], "=>", instruction)
      opcode = (instruction & 0xFF00) >> 12

      # Pre-increment the PC.
      cpu_registers[PC] += 2

      if stop_on_zero and instruction == 0x0:
        print("Stopping on zeroed instruction!")
        exit_reason[0] = EXIT_TERMINATION
        break

      # 00E0 - CLS
      if instruction == 0x00E0:
        for i in range(DISPLAY_WIDTH * DISPLAY_HEIGHT):
          display[i] = 0

      # 00EE - RET
      elif instruction == 0x00EE:
        cpu_registers[PC] = stack[cpu_registers[SP]]
        cpu_registers[SP] -= 1

      # 00FD - EXIT
      elif instruction == 0x00FD:
        exit_reason[0] = EXIT_TERMINATION
        break

      # 1nnn - JP addr.
      elif opcode == 0x1:
        addr = instruction & 0x0FFF
        if cpu_registers[PC] - 2 == addr and stop_on_infinite_loop:
          print("Stopping on infinite loop!")
          exit_reason[0] = EXIT_TERMINATION
          break
        cpu_registers[PC] = addr

      # 2nnn - CALL addr
      elif opcode == 0x2:
        addr = instruction & 0x0FFF
        stack[cpu_registers[SP]] = cpu_registers[PC]
        cpu_registers[SP] += 1
        cpu_registers[PC] = addr

      # 3xkk - SE Vx, byte
      elif opcode == 0x3:
        reg = (instruction & 0x0F00) >> 8
        value = instruction & 0x00FF
        if gen_registers[reg] == value:
          cpu_registers[PC] += 2

      # 4xkk - SNE Vx, byte
      elif opcode == 0x4:
        reg = (instruction & 0x0F00) >> 8
        value = instruction & 0x00FF
        if gen_registers[reg] != value:
          cpu_registers[PC] += 2

      # 5xy0 - SE Vx, Vy
      elif opcode == 0x5:
        reg1 = (instruction & 0x0F00) >> 8
        reg2 = (instruction & 0x00F0) >> 4
        if gen_registers[reg1] == gen_registers[reg2]:
          cpu_registers[PC] += 2

      # 6xkk - LD Vx, byte
      elif opcode == 0x6:
        reg = (instruction & 0x0F00) >> 8
        value = instruction & 0x00FF
        gen_registers[reg] = value

      # 7xkk - ADD Vx, byte
      elif opcode == 0x7:
        reg = (instruction & 0x0F00) >> 8
        value = instruction & 0x00FF
        gen_registers[reg] += value

      # 8... - Many
      elif opcode == 0x8:
        subcode = instruction & 0x000F
        reg1 = (instruction & 0x0F00) >> 8
        reg2 = (instruction & 0x00F0) >> 4

        # 8xy0 - LD Vx, Vy
        if subcode == 0x0:
          gen_registers[reg1] = gen_registers[reg2]

        # 8xy1 - OR Vx, Vy
        elif subcode == 0x1:
          gen_registers[reg1] |= gen_registers[reg2]

        # 8xy2 - AND Vx, Vy
        elif subcode == 0x2:
          gen_registers[reg1] &= gen_registers[reg2]

        # 8xy3 - XOR Vx, Vy
        elif subcode == 0x3:
          gen_registers[reg1] ^= gen_registers[reg2]

        # 8xy4 - ADD Vx, Vy
        elif subcode == 0x4:
          gen_registers[VF] = 1 if (gen_registers[reg1] + gen_registers[reg2]) > 255 else 0
          gen_registers[reg1] += gen_registers[reg2]

        # 8xy5 - SUB Vx, Vy
        elif subcode == 0x5:
          if sub_vf_compat:
            gen_registers[VF] = 1 if gen_registers[reg1] >= gen_registers[reg2] else 0
          else:
            gen_registers[VF] = 1 if gen_registers[reg1] > gen_registers[reg2] else 0
          gen_registers[reg1] -= gen_registers[reg2]

        # 8xy6 - SHR Vx {, Vy}
        elif subcode == 0x6:
          gen_registers[VF] = (gen_registers[reg1] & 0x01)
          gen_registers[reg1] >>= 1

        # 8xy7 - SUBN Vx, Vy
        elif subcode == 0x7:
          if sub_vf_compat:
            gen_registers[VF] = 1 if gen_registers[reg2] >= gen_registers[reg1] else 0
          else:
            gen_registers[VF] = 1 if gen_registers[reg2] > gen_registers[reg1] else 0
          gen_registers[reg1] = gen_registers[reg2] - gen_registers[reg1]

        # 8xyE - SHL Vx {, Vy}
        elif subcode == 0xE:
          gen_registers[VF] = 1 if (gen_registers[reg1] & 0x80 != 0) else 0
          gen_registers[reg1] <<= 1

        # Unknown
        else:
          print("Unknown subinstruction", opcode, subcode, instruction)

      # 9xy0 - SNE Vx, Vy
      elif opcode == 0x9:
        vx = (instruction & 0x0F00) >> 8
        vy = (instruction & 0x00F0) >> 4
        if gen_registers[vx] != gen_registers[vy]:
          cpu_registers[PC] += 2

      # Annn - LD I, addr
      elif opcode == 0xA:
        addr = instruction & 0x0FFF
        cpu_registers[I] = addr

      # Bnnn - JP V0, addr
      elif opcode == 0xB:
        addr = instruction & 0x0FFF
        cpu_registers[PC] = addr + gen_registers[V0]

      # Cxkk - RND Vx, byte
      elif opcode == 0xC:
        vx = (instruction & 0x0F00) >> 8
        value = instruction & 0x00FF
        random_val = xoroshiro128p_uniform_float32(rng, cuda.grid(1))
        gen_registers[vx] = int(random_val * 256) & value

      # Dxyn - DRW Vx, Vy, nibble
      elif opcode == 0xD:
        vx = (instruction & 0x0F00) >> 8
        vy = (instruction & 0x00F0) >> 4
        nibble = instruction & 0x000F

        # TODO: wrap.
        for row in range(nibble):
          line = memory[cpu_registers[I] + row]
          y = (gen_registers[vy] + row) & (DISPLAY_HEIGHT - 1)
          for col in range(8):
            pixel = 1 if (line & (0x80 >> col)) != 0 else 0
            x = (gen_registers[vx] + col) & (DISPLAY_WIDTH - 1)
            
            pos = DISPLAY_WIDTH * y + x
            display[pos] ^= pixel
            if display[pos] == 0 and gen_registers[VF] == 0:
              gen_registers[VF] == 1
        exit_reason[0] = EXIT_REDRAW
        break

      # E...
      elif opcode == 0xE:
        subcode = instruction & 0x00FF
        reg = (instruction & 0x0F00) >> 8

        # Ex9E - SKP Vx
        if subcode == 0x9E:
          print("Not implemented: SKP")
          pass

        # ExA1 - SKNP Vx
        elif subcode == 0xA1:
          print("Not implemented: SKNP")
          pass

        # Unknown
        else:
          print("Unknown subinstruction", opcode, subcode, instruction)

      # F...
      elif opcode == 0xF:
        subcode = instruction & 0x00FF
        reg = (instruction & 0x0F00) >> 8

        # Fx07 - LD Vx, DT
        if subcode == 0x07:
          gen_registers[reg] = cpu_registers[DT]
          pass

        # Fx0A - LD Vx, K
        elif subcode == 0x0A:
          print("Not implemented: LD Vx, K")
          pass

        # Fx15 - LD DT, Vx
        elif subcode == 0x15:
          cpu_registers[DT] = gen_registers[reg]
          pass

        # Fx18 - LD ST, Vx
        elif subcode == 0x18:
          print("Not implemented: LD ST, Vx")
          pass

        # Fx1E - ADD I, Vx
        elif subcode == 0x1E:
          gen_registers[VF] = 1 if cpu_registers[I] + gen_registers[reg] > 255 else 0
          cpu_registers[I] = cpu_registers[I] + gen_registers[reg]

        # Fx29 - LD F, Vx
        elif subcode == 0x29:
          val = (gen_registers[reg] & 0x0F)
          cpu_registers[I] = 5*val

        # Fx33 - LD B, Vx
        elif subcode == 0x33:
          memory[cpu_registers[I]] = gen_registers[reg] / 100 % 10
          memory[cpu_registers[I] + 0x1] = gen_registers[reg] / 10 % 10
          memory[cpu_registers[I] + 0x2] = gen_registers[reg] % 10

        # Fx55 - LD [I], Vx
        elif subcode == 0x55:
          for reg_i in range(reg+1):
            memory[cpu_registers[I] + reg_i] = gen_registers[reg_i]

        # Fx65 - LD Vx, [I]
        elif subcode == 0x65:
          for reg_i in range(reg+1):
            gen_registers[reg_i] = memory[cpu_registers[I] + reg_i]
        
        # Fx75 - LD R, Vx
        elif subcode == 0x75:
          for reg_i in range(reg+1):
            hp_flags[reg_i] = gen_registers[reg_i]

        # Fx85 - LD R, Vx
        elif subcode == 0x85:
          for reg_i in range(reg+1):
            gen_registers[reg_i] = hp_flags[reg_i]

        # Unknown
        else:
          print("Unknown subinstruction", opcode, subcode, instruction)

      # Unknown
      else:
        if opcode != 0x0:
          print("Unknown instruction", opcode, instruction)

      if cpu_registers[PC] > 0x0FFF:
        print("WARN: end of memory!")
        exit_reason[0] = EXIT_TERMINATION
        break
