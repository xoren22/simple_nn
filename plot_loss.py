import time
import numpy as np
import matplotlib.pyplot as plt

class plot_loss:
	def __init__(self, num_epochs, num_plots):
		self.num_plots = num_plots
		self.num_epochs = num_epochs

		self.metrices = [[0] for _ in range(num_plots)]

		self.fig = plt.figure()
		self.axarr = [self.fig.add_subplot(211), self.fig.add_subplot(212)]
		self.lines = [ax.plot(range(num_epochs), [0]*num_epochs, 'r-')[0] for ax in self.axarr]     

	def update_plot(self, new_metrics):
		# plt.ion()
		for i in range(len(new_metrics)):
			self.metrices[i].append(new_metrics[i])

			[l.set_ydata(self.metrices[i]+[0]*(self.num_epochs-i)) for l in self.lines]

			self.fig.canvas.draw()
			self.fig.canvas.flush_events()


p = plot_loss(600, 2)
for i in range(10):
	time.sleep(0.2)
	p.update_plot(list(np.random.uniform(-1, 1, 2)))