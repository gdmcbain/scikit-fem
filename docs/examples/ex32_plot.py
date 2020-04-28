from vtkplotter import Plotter, screenshot

Plotter(offscreen=True).load("ex32.vtk").pointColors("pressure").show()
screenshot("ex32.png")
