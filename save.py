

def write_file(filess, T):
	
	f = open(filess, "w")
	for o in T:
		f.write("[\n")
		for l in o:
			f.write(str(l)+"\n")
		f.write("]\n")
	f.close()


def save_hidden_weight(nb_hidden, hiddenw):

	for i in range(nb_hidden):
		write_file("save/base_nn_hid_" + str(i+1) + "w.nn", hiddenw[i])


def load_hiddenw(filess, hiddenw):

	f = open(filess, "r")
	s = f.read().splitlines()
	h = 0
	
	for o in s:
		#print(o)
		if o == "[":
			h = []
		elif o == "]":
			hiddenw.append(h)
		else:
			h.append(float(o))
			

def load_hidden_weight(hiddenw, nb_hidden):

	for i in range(nb_hidden):
		hiddenw.append([])
		load_hiddenw("save/base_nn_hid_" + str(i+1) + "w.nn", hiddenw[i])

def load_hidden_weight_v(hiddenw, nb_hidden):

	for i in range(nb_hidden):
		hiddenw.append([])
		load_hiddenw("valid/NN/base_nn_hid_" + str(i+1) + "w.nn", hiddenw[i])



def display_hidden_weight(hiddenw, nb_hidden):

	
	for i in range(nb_hidden):
		for j in hiddenw[i]:
			print("------------------------------------")
			I = 0
			for k in j:
				print(k)
				I+=1
				if I > 3:
					break
					
					
def write_fileb(filess, T):
	
	f = open(filess, "w")
	for o in T:
		f.write(str(o)+"\n")
		
	f.close()


def save_hidden_bias(nb_hidden, hiddenb):

	for i in range(nb_hidden):
		write_fileb("save/base_nn_hid_" + str(i+1) + "b.nn", hiddenb[i])
	
def load_hiddenb(filess, hiddenb):

	f = open(filess, "r")
	s = f.read().splitlines()
		
	for o in s:
		hiddenb.append(float(o))
						
def load_hidden_bias(hiddenb, nb_hidden):

	for i in range(nb_hidden):
		hiddenb.append([])
		load_hiddenb("save/base_nn_hid_" + str(i+1) + "b.nn", hiddenb[i])

def load_hidden_bias_v(hiddenb, nb_hidden):

	for i in range(nb_hidden):
		hiddenb.append([])
		load_hiddenb("valid/NN/base_nn_hid_" + str(i+1) + "b.nn", hiddenb[i])



def display_hidden_bias(hiddenb, nb_hidden):
	
	for i in range(nb_hidden):
		print("------------------------------------")
		for j in hiddenb[i]:
			print(j)


def save_output_weight(outputw):

	write_file("save/base_nn_out_w.nn", outputw[0])


def load_output_weight(outputw):

	outputw.append([])
	load_hiddenw("save/base_nn_out_w.nn", outputw[0])

def load_output_weight_v(outputw):

	outputw.append([])
	load_hiddenw("valid/NN/base_nn_out_w.nn", outputw[0])


def save_output_bias(outputb):

	write_fileb("save/base_nn_out_b.nn", outputb[0])
		

def load_output_bias(outputb):

	outputb.append([])
	load_hiddenb("save/base_nn_out_b.nn", outputb[0])

def load_output_bias_v(outputb):

	outputb.append([])
	load_hiddenb("valid/NN/base_nn_out_b.nn", outputb[0])

	
		


