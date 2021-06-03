# CHIP8 on CUDA

This is an interpreter for the [CHIP-8](https://en.wikipedia.org/wiki/CHIP-8) CPU that runs (almost) entirely on a [CUDA](https://en.wikipedia.org/wiki/CUDA) kernel. Simple as that.

## Disclaimer

This is literally the first time I'm working with CUDA (besides cryptomining). Be aware that the code might be very, very bad, as well as the performance. The way I'm moving memory in and out of the GPU is slow and there are probably better ways of doing so. Also, Python might screw up the performance a bit.

I don't think I have to say this, but don't use this in any kind of production environment.

## Background

I was on vacation (at home due to the pandemic outside) and was wondering what can a GPU do (besides math). Computers are basically powerful calculators, so I was wondering if I could emulate another architecture inside a GPU. I thought at first about targeting a GameBoy CPU, or maybe an AVR which I'm familiar with, but they require some I/O i wasn't willing to implement. So I googled for "the simplest CPU to create an emulator for" (for real), and CHIP-8 came up.

The implementation is heavily based on [Cowgod's Chip-8 Technical Reference](http://devernay.free.fr/hacks/chip8/C8TECH10.HTM) and [Werner Stoop's implementation](https://github.com/wernsey/chip8/blob/master/chip8.c).

## Usage

Assuming you want to run this, all you gotta pass as argument is the path of a .ch8 file (CHIP-8 binary code). The code requires Numba, NumPy and SDL2 (for the display) to work. It has a few flags that help debugging and dealing with weird code, such as exiting on `0x0000`, infinite loops and even `VF` compatibility mode for some test programs.

Right now there are a few test programs that more or less run on it:

1. [BC_TEST](https://github.com/stianeklund/chip8/blob/master/roms/BC_test.ch8) (passes)
2. [C8_TEST](https://github.com/Skosulor/c8int/blob/master/test/c8_test.c8) (error 16)
3. [chip8-test-rom](https://github.com/corax89/chip8-test-rom) (runs fine)
4. [SCTEST](https://github.com/daniel5151/AC8E/blob/master/roms/SCTEST) (error)

And a few programs that actually load:

1. [MAZE](https://github.com/daniel5151/AC8E/blob/master/roms/games/MAZE) (runs fine)
2. [PONG](https://github.com/daniel5151/AC8E/blob/master/roms/games/PONG) (runs fine but does nothing)

The ROMs above have links to the repository I got them from if you want a copy.

## Known bugs and limitations

1. Keyboard is not implemented yet, so keyboard-related instructions will fail.
2. Timer/delays/sound/etc do nothing. They still need to be coded.
2. Display has a single mode (the original).
3. RND crashes the CUDA interface when using a device different than 0. Seems to be a bug on Numba actually.
4. Code runs on a single thread (I guess), so the GPU is just an expensive single-core CPU.
5. The SDL code is messy and the performance is bad.
