import numpy
import matplotlib.pyplot as plt
import threading
import sys

import sdl2
from sdl2 import render
import sdl2.ext

from core import Core, DISPLAY_HEIGHT, DISPLAY_WIDTH

PIXEL_MULT = 8

class CoreThread(threading.Thread):
  def __init__(self, program, redraw_func):
    threading.Thread.__init__(self)
    self._program = program
    self._redraw_func = redraw_func


  def run(self):
    # NOTE: using a cuda_device different than 0 sometimes fucks the RND up.
    self._core = Core(cuda_device=0, sub_vt_compat=False)
    self._core.load(self._program)
    self._core.run(lambda display: self._redraw_func(display))
    print("WARN: core exited!")
  
  def dump(self):
    self._core.dump()

WHITE = sdl2.ext.Color(255, 255, 255)
BLACK = sdl2.ext.Color(0, 0, 0)

def redraw(display_data):
  global renderer

  for x in range(DISPLAY_WIDTH):
    for y in range(DISPLAY_HEIGHT):
      pos = DISPLAY_WIDTH * y + x
      color = WHITE if display_data[pos] == 1 else BLACK
      renderer.draw_rect([x*PIXEL_MULT,y*PIXEL_MULT,PIXEL_MULT,PIXEL_MULT], color)
  renderer.present()


if len(sys.argv) != 2:
  print(sys.argv[0], "<program.ch8>")
  sys.exit(1)


sdl2.ext.init()
window = sdl2.ext.Window("Test", size=(DISPLAY_WIDTH * PIXEL_MULT, DISPLAY_HEIGHT * PIXEL_MULT))
renderer = sdl2.ext.Renderer(window)
window.show()

with open(sys.argv[1], "rb") as f:
  program = f.read()

core = CoreThread(program, lambda display: redraw(display))
core.start()

running = True
while running:
  events = sdl2.ext.get_events()
  for event in events:
    if event.type == sdl2.SDL_QUIT:
      running = False
      core.join()
      core.dump()
      break
  window.refresh()
