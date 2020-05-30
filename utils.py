import numpy as np
import matplotlib.pyplot as plt

def to_categorical(y):
    uniques = np.unique(y)
    sparse_cat = np.zeros((len(y), np.max(y)+1))
    sparse_cat[np.arange(len(y)), y] = True
    res = sparse_cat[:, uniques]
    
    return res

def dist(x, y, title=None):
	x = x.reshape(-1)
	plt.hist(x[y[:,0]], bins=50, alpha=.5)
	plt.hist(x[y[:,1]], bins=50, alpha=.5)
	if title:
		plt.title(title)
	plt.show()



class LogPrint:
	def __init__(self, metic_names, flush=True):
		self.first = True
		self.flush = flush
		self.metic_names = metic_names
		self.metic_name_lenghts = [len(m) for m in self.metic_names]
		self.col_dist = int(max(self.metic_name_lenghts) * 2)


	def print(self, new_metrics):
		if self.first:
			formated_names = "\n"*20 + "-"*70+"\n"
			for i, name in enumerate(self.metic_names):
				formated_names += name + (self.col_dist-self.metic_name_lenghts[i])*" "
			self.first = False

			formated_names += "  |\n" + "-"*70 + "\n"
			print(formated_names)

		formated_vals = ""
		for i, val in enumerate(new_metrics):
			formated_vals += "%5f"%val + (self.col_dist-len("%5f"%val))*" "
		print(formated_vals)


if __name__ == '__main__':
	log = LogPrint(["loss", "accurecy"])
	for i in range(4):
		fake_loss, fake_acc = np.random.uniform(0,3), np.random.uniform(0,100)
		log.print([fake_loss, fake_acc])
