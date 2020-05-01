import matplotlib.pyplot as plt
import numpy as np

class plot_loss:
	def __init__(self):
		plt.ion()
		fig = plt.figure()
		ax = fig.add_subplot(111)

	def update_plot(self):
		line1, = ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma

		for phase in np.linspace(0, 10*np.pi, 500):
		    line1.set_ydata(np.sin(x + phase))
		    fig.canvas.draw()
		    fig.canvas.flush_events()