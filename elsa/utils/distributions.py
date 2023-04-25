""" Distribution class """

from elsa.utils.plots import *
from elsa.utils.observables import Observable

import numpy as np

import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class Distribution(Observable):
	"""Custom Distribution class.

	Defines which Observables will be plotted depending on the
	specified dataset.
	"""
	def __init__(self,
				 real_data: np.ndarray,
				 gen_data: np.ndarray,
				 name: str,
				 label_name: str,
				 log_dir: str,
				 dataset: str,
				 latent: bool=False,
				 weights: np.ndarray=None,
				 extra_data: np.ndarray=None
     	):
		super(Distribution, self).__init__()
		self.real_data = real_data
		self.gen_data = gen_data
		self.name = name
		self.label_name = label_name
		self.log_dir = log_dir
		self.dataset = dataset
		self.latent = latent
		self.weights = weights
		self.extra_data = extra_data

	def plot(self):
		if self.latent == True:
			self.latent_distributions()
		else:
			if self.dataset == 'wp_2j':
				self.w_2jets_distributions()
			elif self.dataset == 'wp_3j':
				self.w_3jets_distributions()
			elif self.dataset == 'wp_4j':
				self.w_4jets_distributions()
			elif self.dataset == '2d_ring_gaussian':
				self.basic_2d_distributions()
			elif self.dataset == 'annulus':
				self.basic_2d_distributions()
			elif self.dataset == 'pinwheel':
				self.pinwheel_distributions()
			elif self.dataset == 'circles':
				self.circles_distributions()
			elif self.dataset == 'eight_gaussians':
				self.eight_gauss_distributions()
			else:
				self.basic_2d_distributions()

		# pylint: disable=W0702
		try:
			plt.rc("text", usetex=True)
			plt.rc("font", family="serif")
			plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
		except:
			print("No latex installed")

		if True:
			with PdfPages(self.log_dir + '/' + self.dataset + '_' + self.name + '_ratio.pdf') as pp:
				for observable in self.args.keys():
					fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios' : [4, 1], 'hspace' : 0.00})
					plot_distribution_ratio(fig, axs, self.real_data, self.gen_data, self.label_name, self.args[observable], self.weights, self.extra_data)
					fig.savefig(pp, bbox_inches='tight', format='pdf', pad_inches=0.05)
					plt.close()

		if False:
			with PdfPages(self.log_dir + '/' + self.dataset + '_' + self.name + '_2d.pdf') as pp:
				for i, observable in enumerate(list(self.args2.keys())):
					for observable2 in list(self.args2.keys())[i+1:]:
						fig, axs = plt.subplots(1,3, figsize=(20,5))
						plot_2d_distribution(fig, axs, self.real_data, self.gen_data, self.args2[observable], self.args2[observable2], self.weights, self.extra_data)
						plt.subplots_adjust(wspace=0.45, hspace=0.25)
						fig.savefig(pp, bbox_inches='tight', format='pdf', pad_inches=0.05)
						plt.close()

	def basic_2d_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			'x' : ([0], self.coord_0, 100, (-1.2,1.2) ,r'$x$', r'$x$',False),
			'y' : ([0], self.coord_1, 100, (-1.2,1.2) ,r'$y$', r'$y$',False),
		}	 

		args2 = {			 
			'x' : ([0], self.coord_0, 100, (-0.6,0.6) ,r'$x$', r'$x$',False),
			'y' : ([0], self.coord_1, 100, (-0.6,0.6) ,r'$y$', r'$y$',False),
		}
	 
		self.args = args
		self.args2 = args2

	def pinwheel_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			'x' : ([0], self.coord_0, 100, (-3.5,3.5) ,r'$x$', r'$x$',False),
			'y' : ([1], self.coord_1, 100, (-3.5,3.5) ,r'$y$', r'$y$',False),
		}	 

		args2 = {			 
			'x' : ([0], self.coord_0, 200, (-3.5,3.5) ,r'$x$', r'$x$',False),
			'y' : ([1], self.coord_1, 200, (-3.5,3.5) ,r'$y$', r'$y$',False),
		}
	 
		self.args = args
		self.args2 = args2

	def eight_gauss_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			'x' : ([0], self.coord_0, 200, (-4.1,4.1) ,r'$x$', r'x',False),
			'y' : ([0], self.coord_1, 200, (-4.1,4.1) ,r'$y$', r'y',False),
		}	 

		args2 = {			 
			'x' : ([0], self.coord_0, 200, (-4.1,4.1) ,r'$x$', r'x',False),
			'y' : ([0], self.coord_1, 200, (-4.1,4.1) ,r'$y$', r'y',False),
		}
	 
		self.args = args
		self.args2 = args2

	def circles_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			'x' : ([0], self.coord_0, 100, (-3.5,3.5) ,r'$x$', r'$x$',False),
			'y' : ([1], self.coord_1, 100, (-3.5,3.5) ,r'$y$', r'$y$',False),
		}	 

		args2 = {			 
			'x' : ([0], self.coord_0, 200, (-3.5,3.5) ,r'$x$', r'$x$',False),
			'y' : ([1], self.coord_1, 200, (-3.5,3.5) ,r'$y$', r'$y$',False),
		}
	 
		self.args = args
		self.args2 = args2

	def w_2jets_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			'ptW' : ([0], self.transverse_momentum, 40, (0,300) ,r'$p_{\mathrm{T}, \mathrm{W}}$ [GeV]', r'p_{\mathrm{T}, \mathrm{W}}',False),
			'pxW' : ([0], self.x_momentum, 50, (-160,160), r'$p_{x, \mathrm{W}}$ [GeV]', r'p_{x, \mathrm{W}}',False),
			'pyW' : ([0], self.y_momentum, 50, (-160,160), r'$p_{y, \mathrm{W}}$ [GeV]', r'p_{y, \mathrm{W}}',False),
			'pzW' : ([0], self.z_momentum, 50, (-600,600), r'$p_{z, \mathrm{W}}$ [GeV]', r'p_{z, \mathrm{W}}',False),
			'EW'  : ([0], self.energy, 40, (0,1000), r'$E_{\mathrm{W}}$ [GeV]', r'E_{\mathrm{W}}',False),
			#---------------------#		
			'ptj1' : ([1], self.transverse_momentum, 40, (0,180) ,r'$p_{\mathrm{T}, \mathrm{j}_1}$ [GeV]', r'p_{\mathrm{T}, \mathrm{j}_1}',False),
			'pxj1' : ([1], self.x_momentum, 40, (-120,120), r'$p_{x, \mathrm{j}_1}$ [GeV]', r'p_{x, \mathrm{j}_1}',False),
			'pyj1' : ([1], self.y_momentum, 40, (-120,120), r'$p_{y, \mathrm{j}_1}$ [GeV]', r'p_{y, \mathrm{j}_1}',False),
			'pzj1' : ([1], self.z_momentum, 50, (-400,400), r'$p_{z, \mathrm{j}_1}$ [GeV]', r'p_{z, \mathrm{j}_1}',False),
			'Ej1'  : ([1], self.energy, 40, (0,600), r'$E_{\mathrm{j}_1}$ [GeV]', r'E_{\mathrm{j}_1}',False),
			#---------------------#			
			'ptj2' : ([2], self.transverse_momentum, 40, (0,180) ,r'$p_{\mathrm{T}, \mathrm{j}_2}$ [GeV]', r'p_{\mathrm{T}, \mathrm{j}_2}',False),
			'pxj2' : ([2], self.x_momentum, 40, (-120,120), r'$p_{x, \mathrm{j}_2}$ [GeV]', r'p_{x, \mathrm{j}_2}',False),
			'pyj2' : ([2], self.y_momentum, 40, (-120,120), r'$p_{y, \mathrm{j}_2}$ [GeV]', r'p_{y, \mathrm{j}_2}',False),
			'pzj2' : ([2], self.z_momentum, 50, (-400,400), r'$p_{z, \mathrm{j}_2}$ [GeV]', r'p_{z, \mathrm{j}_2}',False),
			'Ej2'  : ([2], self.energy, 40, (0,600), r'$E_{\mathrm{j}_2}$ [GeV]', r'E_{\mathrm{j}_2}',False),
			#---------------------#
   			'dPhijj' : ([1,2], self.delta_phi, 40, (0,3.14) ,r'$\Delta\phi_{\mathrm{j}\mathrm{j}}$', r'\Delta\phi_{\mathrm{j}\mathrm{j}}',False),
			'dEtajj' : ([1,2], self.delta_rapidity, 40, (0,5) ,r'$\Delta\eta_{\mathrm{j}\mathrm{j}}$', r'\Delta\eta_{\mathrm{j}\mathrm{j}}',False),
			'dRjj' : ([1,2], self.delta_R, 40, (0,8) ,r'$\Delta R_{\mathrm{j}\mathrm{j}}$', r'\Delta R_{\mathrm{j}\mathrm{j}}',False),
		 	'mwjj' : ([0,1,2], self.invariant_mass, 50, (0,2000), r'$M_{\mathrm{W}\mathrm{j}\mathrm{j}}$ [GeV]', r'p_{x, j2}',False),
		}	 

		args2 = {			 
			'ptW' : ([0], self.transverse_momentum, 40, (0,300) ,r'$p_{\mathrm{T}, \mathrm{W}}$ [GeV]', r'p_{\mathrm{T}, \mathrm{W}}',False),
			'EW'  : ([0], self.energy, 40, (0,1000), r'$E_{\mathrm{W}}$ [GeV]', r'E_{\mathrm{W}}',False),
			#---------------------#		
			'ptj1' : ([1], self.transverse_momentum, 40, (0,180) ,r'$p_{\mathrm{T}, \mathrm{j}_1}$ [GeV]', r'p_{\mathrm{T}, \mathrm{j}_1}',False),
			'Ej1'  : ([1], self.energy, 40, (0,600), r'$E_{\mathrm{j}_1}$ [GeV]', r'E_{\mathrm{j}_1}',False),
			#---------------------#			
			'ptj2' : ([2], self.transverse_momentum, 40, (0,180) ,r'$p_{\mathrm{T}, \mathrm{j}_2}$ [GeV]', r'p_{\mathrm{T}, \mathrm{j}_2}',False),
			'Ej2'  : ([2], self.energy, 40, (0,600), r'$E_{\mathrm{j}_2}$ [GeV]', r'E_{\mathrm{j}_2}',False),
			#---------------------#
   			'dPhijj' : ([1,2], self.delta_phi, 40, (0,3.14) ,r'$\Delta\phi_{\mathrm{j}\mathrm{j}}$', r'\Delta\phi_{\mathrm{j}\mathrm{j}}',False),
			'dEtajj' : ([1,2], self.delta_rapidity, 40, (0,5) ,r'$\Delta\eta_{\mathrm{j}\mathrm{j}}$', r'\Delta\eta_{\mathrm{j}\mathrm{j}}',False),
			'dRjj' : ([1,2], self.delta_R, 40, (0,8) ,r'$\Delta R_{\mathrm{j}\mathrm{j}}$', r'\Delta R_{\mathrm{j}\mathrm{j}}',True),
		 	'mjj' : ([1,2], self.invariant_mass, 40, (0,1000), r'$M_{\mathrm{j}\mathrm{j}}$ [GeV]', r'p_{x, j2}',True),
			#---------------------#			
		}	 
	 
		self.args = args
		self.args2 = args2

	def w_3jets_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			'ptW' : ([0], self.transverse_momentum, 40, (0,300) ,r'$p_{\mathrm{T}, \mathrm{W}}$ [GeV]', r'p_{\mathrm{T}, \mathrm{W}}',False),
			'pxW' : ([0], self.x_momentum, 50, (-160,160), r'$p_{x, \mathrm{W}}$ [GeV]', r'p_{x, \mathrm{W}}',False),
			'pyW' : ([0], self.y_momentum, 50, (-160,160), r'$p_{y, \mathrm{W}}$ [GeV]', r'p_{y, \mathrm{W}}',False),
			'pzW' : ([0], self.z_momentum, 50, (-600,600), r'$p_{z, \mathrm{W}}$ [GeV]', r'p_{z, \mathrm{W}}',False),
			'EW'  : ([0], self.energy, 40, (0,1000), r'$E_{\mathrm{W}}$ [GeV]', r'E_{\mathrm{W}}',False),
			'etaW': ([0], self.pseudo_rapidity, 40, (-7,7), r'$\eta_{\mathrm{W}}$', r'\eta_{\mathrm{W}}',False),
			#---------------------#		
			'ptj1' : ([1], self.transverse_momentum, 40, (0,180) ,r'$p_{\mathrm{T}, \mathrm{j}_1}$ [GeV]', r'p_{\mathrm{T}, \mathrm{j}_1}',False),
			'pxj1' : ([1], self.x_momentum, 40, (-120,120), r'$p_{x, \mathrm{j}_1}$ [GeV]', r'p_{x, \mathrm{j}_1}',False),
			'pyj1' : ([1], self.y_momentum, 40, (-120,120), r'$p_{y, \mathrm{j}_1}$ [GeV]', r'p_{y, \mathrm{j}_1}',False),
			'pzj1' : ([1], self.z_momentum, 50, (-400,400), r'$p_{z, \mathrm{j}_1}$ [GeV]', r'p_{z, \mathrm{j}_1}',False),
			'Ej1'  : ([1], self.energy, 40, (0,600), r'$E_{\mathrm{j}_1}$ [GeV]', r'E_{\mathrm{j}_1}',False),
   			'etaj1': ([1], self.pseudo_rapidity, 40, (-7,7), r'$\eta_{\mathrm{j}_1}$', r'\eta_{\mathrm{j}_1}',False),
			#---------------------#			
			'ptj2' : ([2], self.transverse_momentum, 40, (0,180) ,r'$p_{\mathrm{T}, \mathrm{j}_2}$ [GeV]', r'p_{\mathrm{T}, \mathrm{j}_2}',False),
			'pxj2' : ([2], self.x_momentum, 40, (-120,120), r'$p_{x, \mathrm{j}_2}$ [GeV]', r'p_{x, \mathrm{j}_2}',False),
			'pyj2' : ([2], self.y_momentum, 40, (-120,120), r'$p_{y, \mathrm{j}_2}$ [GeV]', r'p_{y, \mathrm{j}_2}',False),
			'pzj2' : ([2], self.z_momentum, 50, (-400,400), r'$p_{z, \mathrm{j}_2}$ [GeV]', r'p_{z, \mathrm{j}_2}',False),
			'Ej2'  : ([2], self.energy, 40, (0,600), r'$E_{\mathrm{j}_2}$ [GeV]', r'E_{\mathrm{j}_2}',False),
			'etaj1': ([2], self.pseudo_rapidity, 40, (-7,7), r'$\eta_{\mathrm{j}_2}$', r'\eta_{\mathrm{j}_2}',False),
			#---------------------#			
			'ptj3' : ([3], self.transverse_momentum, 40, (0,180) ,r'$p_{\mathrm{T}, \mathrm{j}_3}$ [GeV]', r'p_{\mathrm{T}, \mathrm{j}_3}',False),
			'pxj3' : ([3], self.x_momentum, 40, (-120,120), r'$p_{x, \mathrm{j}_3}$ [GeV]', r'p_{x, \mathrm{j}_3}',False),
			'pyj3' : ([3], self.y_momentum, 40, (-120,120), r'$p_{y, \mathrm{j}_3}$ [GeV]', r'p_{y, \mathrm{j}_3}',False),
			'pzj3' : ([3], self.z_momentum, 50, (-400,400), r'$p_{z, \mathrm{j}_3}$ [GeV]', r'p_{z, \mathrm{j}_3}',False),
			'Ej3'  : ([3], self.energy, 40, (0,600), r'$E_{\mathrm{j}_3}$ [GeV]', r'E_{\mathrm{j}_3}',False),
			'etaj3': ([3], self.pseudo_rapidity, 40, (-7,7), r'$\eta_{\mathrm{j}_3}$', r'\eta_{\mathrm{j}_3}',False),
			#---------------------#
   			'dPhij1j2' : ([1,2], self.delta_phi, 40, (0,3.14) ,r'$\Delta\phi_{\mathrm{j}_1\mathrm{j}_2}$', r'\Delta\phi_{\mathrm{j}_1\mathrm{j}_2}',False),
			'dEtaj1j2' : ([1,2], self.delta_rapidity, 40, (0,5) ,r'$\Delta\eta_{\mathrm{j}_1\mathrm{j}_2}$', r'\Delta\eta_{\mathrm{j}_1\mathrm{j}_2}',False),
			'dRj1j2' : ([1,2], self.delta_R, 40, (0,8) ,r'$\Delta R_{\mathrm{j}_1\mathrm{j}_2}$', r'\Delta R_{\mathrm{j}_1\mathrm{j}_2}',False),
			#---------------------#
			'dPhij1j3' : ([1,3], self.delta_phi, 40, (0,3.14) ,r'$\Delta\phi_{\mathrm{j}_1\mathrm{j}_3}$', r'\Delta\phi_{\mathrm{j}_1\mathrm{j}_3}',False),
			'dEtaj1j3' : ([1,3], self.delta_rapidity, 40, (0,5) ,r'$\Delta\eta_{\mathrm{j}_1\mathrm{j}_3}$', r'\Delta\eta_{\mathrm{j}_1\mathrm{j}_3}',False),
			'dRj1j3' : ([1,3], self.delta_R, 40, (0,8) ,r'$\Delta R_{\mathrm{j}_1\mathrm{j}_3}$', r'\Delta R_{\mathrm{j}_1\mathrm{j}_3}',False),
			#---------------------#	
			'dPhij2j3' : ([2,3], self.delta_phi, 40, (0,3.14) ,r'$\Delta\phi_{\mathrm{j}_2\mathrm{j}_3}$', r'\Delta\phi_{\mathrm{j}_2\mathrm{j}_3}',False),
			'dEtaj2j3' : ([2,3], self.delta_rapidity, 40, (0,5) ,r'$\Delta\eta_{\mathrm{j}_2\mathrm{j}_3}$', r'\Delta\eta_{\mathrm{j}_2\mathrm{j}_3}',False),
			'dRj2j3' : ([2,3], self.delta_R, 40, (0,8) ,r'$\Delta R_{\mathrm{j}_2\mathrm{j}_3}$', r'\Delta R_{\mathrm{j}_2\mathrm{j}_3}',False),
			#---------------------#	
		} 

		args2 = {			 
			# 'ptW' : ([0], self.transverse_momentum, 40, (0,300) ,r'$p_{\mathrm{T}, \mathrm{W}}$ [GeV]', r'p_{\mathrm{T}, \mathrm{W}}',False),
			# 'EW'  : ([0], self.energy, 40, (0,1000), r'$E_{\mathrm{W}}$ [GeV]', r'E_{\mathrm{W}}',False),
			# #---------------------#		
			# 'ptj1' : ([1], self.transverse_momentum, 40, (0,180) ,r'$p_{\mathrm{T}, \mathrm{j}_1}$ [GeV]', r'p_{\mathrm{T}, \mathrm{j}_1}',False),
			# 'Ej1'  : ([1], self.energy, 40, (0,600), r'$E_{\mathrm{j}_1}$ [GeV]', r'E_{\mathrm{j}_1}',False),
			# #---------------------#			
			# 'ptj2' : ([2], self.transverse_momentum, 40, (0,180) ,r'$p_{\mathrm{T}, \mathrm{j}_2}$ [GeV]', r'p_{\mathrm{T}, \mathrm{j}_2}',False),
			# 'Ej2'  : ([2], self.energy, 40, (0,600), r'$E_{\mathrm{j}_2}$ [GeV]', r'E_{\mathrm{j}_2}',False),
			# #---------------------#			
			# 'ptj3' : ([3], self.transverse_momentum, 40, (0,180) ,r'$p_{\mathrm{T}, \mathrm{j}_3}$ [GeV]', r'p_{\mathrm{T}, \mathrm{j}_3}',False),
			# 'Ej3'  : ([3], self.energy, 40, (0,600), r'$E_{\mathrm{j}_3}$ [GeV]', r'E_{\mathrm{j}_3}',False),
			#---------------------#
   			'dPhij1j2' : ([1,2], self.delta_phi, 40, (-3.14,3.14) ,r'$\Delta\phi_{\mathrm{j}_1\mathrm{j}_2}$', r'\Delta\phi_{\mathrm{j}_1\mathrm{j}_2}',False),
			'dEtaj1j2' : ([1,2], self.delta_rapidity, 40, (-3,3) ,r'$\Delta\eta_{\mathrm{j}_1\mathrm{j}_2}$', r'\Delta\eta_{\mathrm{j}_1\mathrm{j}_2}',False),
			'dRj1j2' : ([1,2], self.delta_R, 40, (0,8) ,r'$\Delta R_{\mathrm{j}_1\mathrm{j}_2}$', r'\Delta R_{\mathrm{j}_1\mathrm{j}_2}',False),
			#---------------------#
			'dPhij1j3' : ([1,3], self.delta_phi, 40, (-3.14,3.14) ,r'$\Delta\phi_{\mathrm{j}_1\mathrm{j}_3}$', r'\Delta\phi_{\mathrm{j}_1\mathrm{j}_3}',False),
			'dEtaj1j3' : ([1,3], self.delta_rapidity, 40, (-3,3) ,r'$\Delta\eta_{\mathrm{j}_1\mathrm{j}_3}$', r'\Delta\eta_{\mathrm{j}_1\mathrm{j}_3}',False),
			'dRj1j3' : ([1,3], self.delta_R, 40, (0,8) ,r'$\Delta R_{\mathrm{j}_1\mathrm{j}_3}$', r'\Delta R_{\mathrm{j}_1\mathrm{j}_3}',False),
			#---------------------#	
			'dPhij2j3' : ([2,3], self.delta_phi, 40, (-3.14,3.14) ,r'$\Delta\phi_{\mathrm{j}_2\mathrm{j}_3}$', r'\Delta\phi_{\mathrm{j}_2\mathrm{j}_3}',False),
			'dEtaj2j3' : ([2,3], self.delta_rapidity, 40, (-3,3) ,r'$\Delta\eta_{\mathrm{j}_2\mathrm{j}_3}$', r'\Delta\eta_{\mathrm{j}_2\mathrm{j}_3}',False),
			'dRj2j3' : ([2,3], self.delta_R, 40, (0,8) ,r'$\Delta R_{\mathrm{j}_2\mathrm{j}_3}$', r'\Delta R_{\mathrm{j}_2\mathrm{j}_3}',False),
			#---------------------#				
		}	 
	 
		self.args = args
		self.args2 = args2

	def w_4jets_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			'ptW' : ([0], self.transverse_momentum, 40, (0,300) ,r'$p_{T, W}$ [GeV]', r'p_{T, W}',False),
		 	'pxW' : ([0], self.x_momentum, 50, (-160,160), r'$p_{\mathrm{x}, W}$ [GeV]', r'p_{x, W}',False),
			'pyW' : ([0], self.y_momentum, 50, (-160,160), r'$p_{\mathrm{y}, W}$ [GeV]', r'p_{y, W}',False),
			'pzW' : ([0], self.z_momentum, 50, (-600,600), r'$p_{\mathrm{z}, W}$ [GeV]', r'p_{z, W}',False),
			'EW'  : ([0], self.energy, 40, (0,1000), r'$E_{W}$ [GeV]', r'E_{W}',False),
			#---------------------#		
			'ptj1' : ([1], self.transverse_momentum, 40, (0,180) ,r'$p_{T, j1}$ [GeV]', r'p_{T, j1}',False),
		 	'pxj1' : ([1], self.x_momentum, 40, (-120,120), r'$p_{\mathrm{x}, j1}$ [GeV]', r'p_{x, j1}',False),
			'pyj1' : ([1], self.y_momentum, 40, (-120,120), r'$p_{\mathrm{y}, j1}$ [GeV]', r'p_{y, j1}',False),
			'pzj1' : ([1], self.z_momentum, 50, (-400,400), r'$p_{\mathrm{z}, j1}$ [GeV]', r'p_{z, j1}',False),
			'Ej1'  : ([1], self.energy, 40, (0,600), r'$E_{j1}$ [GeV]', r'E_{j1}',False),
			#---------------------#			
			'ptj2' : ([2], self.transverse_momentum, 40, (0,180) ,r'$p_{T, j2}$ [GeV]', r'p_{T, j2}',False),
		 	'pxj2' : ([2], self.x_momentum, 40, (-120,120), r'$p_{\mathrm{x}, j2}$ [GeV]', r'p_{x, j2}',False),
			'pyj2' : ([2], self.y_momentum, 40, (-120,120), r'$p_{\mathrm{y}, j2}$ [GeV]', r'p_{y, j2}',False),
			'pzj2' : ([2], self.z_momentum, 50, (-400,400), r'$p_{\mathrm{z}, j2}$ [GeV]', r'p_{z, j2}',False),
			'Ej2'  : ([2], self.energy, 40, (0,600), r'$E_{j2}$ [GeV]', r'E_{j2}',False),
			#---------------------#			
			'ptj3' : ([3], self.transverse_momentum, 40, (0,180) ,r'$p_{T, j3}$ [GeV]', r'p_{T, j3}',False),
		 	'pxj3' : ([3], self.x_momentum, 40, (-120,120), r'$p_{\mathrm{x}, j3}$ [GeV]', r'p_{x, j3}',False),
			'pyj3' : ([3], self.y_momentum, 40, (-120,120), r'$p_{\mathrm{y}, j3}$ [GeV]', r'p_{y, j3}',False),
			'pzj3' : ([3], self.z_momentum, 50, (-400,400), r'$p_{\mathrm{z}, j3}$ [GeV]', r'p_{z, j3}',False),
			'Ej3'  : ([3], self.energy, 40, (0,600), r'$E_{j3}$ [GeV]', r'E_{j3}',False),
			#---------------------#			
			'ptj4' : ([4], self.transverse_momentum, 40, (0,180) ,r'$p_{T, j4}$ [GeV]', r'p_{T, j4}',False),
		 	'pxj4' : ([4], self.x_momentum, 40, (-120,120), r'$p_{\mathrm{x}, j4}$ [GeV]', r'p_{x, j4}',False),
			'pyj4' : ([4], self.y_momentum, 40, (-120,120), r'$p_{\mathrm{y}, j4}$ [GeV]', r'p_{y, j4}',False),
			'pzj4' : ([4], self.z_momentum, 50, (-400,400), r'$p_{\mathrm{z}, j4}$ [GeV]', r'p_{z, j4}',False),
			'Ej4'  : ([4], self.energy, 40, (0,600), r'$E_{j4}$ [GeV]', r'E_{j4}',False),
			#---------------------#			
		}	 

		args2 = {			 
			'ptW' : ([0], self.transverse_momentum, 40, (0,300) ,r'$p_{T, W}$ [GeV]', r'p_{T, W}',False),
		 	'pxW' : ([0], self.x_momentum, 50, (-160,160), r'$p_{\mathrm{x}, W}$ [GeV]', r'p_{x, W}',False),
			'pyW' : ([0], self.y_momentum, 50, (-160,160), r'$p_{\mathrm{y}, W}$ [GeV]', r'p_{y, W}',False),
			#'pzW' : ([0], self.z_momentum, 50, (-400,400), r'$p_{\mathrm{z}, W}$ [GeV]', r'p_{z, W}',False),
			#'EW'  : ([0], self.energy, 40, (0,1000), r'$E_{W}$ [GeV]', r'E_{W}',False),
			#---------------------#		
			'ptj1' : ([1], self.transverse_momentum, 40, (0,180) ,r'$p_{T, j1}$ [GeV]', r'p_{T, j1}',False),
		 	'pxj1' : ([1], self.x_momentum, 40, (-120,120), r'$p_{\mathrm{x}, j1}$ [GeV]', r'p_{x, j1}',False),
			'pyj1' : ([1], self.y_momentum, 40, (-120,120), r'$p_{\mathrm{y}, j1}$ [GeV]', r'p_{y, j1}',False),
			#'pzj1' : ([1], self.z_momentum, 50, (-400,400), r'$p_{\mathrm{z}, j1}$ [GeV]', r'p_{z, j1}',False),
			#'Ej1'  : ([1], self.energy, 40, (0,600), r'$E_{j1}$ [GeV]', r'E_{j1}',False),
			#---------------------#			
			#'ptj2' : ([2], self.transverse_momentum, 40, (0,180) ,r'$p_{T, j2}$ [GeV]', r'p_{T, j2}',False),
		 	#'pxj2' : ([2], self.x_momentum, 40, (-120,120), r'$p_{\mathrm{x}, j2}$ [GeV]', r'p_{x, j2}',False),
			#'pyj2' : ([2], self.y_momentum, 40, (-120,120), r'$p_{\mathrm{y}, j2}$ [GeV]', r'p_{y, j2}',False),
			#'pzj2' : ([2], self.z_momentum, 50, (-400,400), r'$p_{\mathrm{z}, j2}$ [GeV]', r'p_{z, j2}',False),
			#'Ej2'  : ([2], self.energy, 40, (0,600), r'$E_{j2}$ [GeV]', r'E_{j2}',False),
			#---------------------#			
		}	 
	 
		self.args = args
		self.args2 = args2

	def latent_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			'z0' : ([0], self.coord_0, 60, (0,1) ,r'$z_0$', r'$z_0$',False),
			'z1' : ([1], self.coord_1, 60, (0,1) ,r'$z_1$', r'$z_1$',False),
			'z2' : ([2], self.coord_2, 60, (0,1) ,r'$z_2$', r'$z_2$',False),
			'z3' : ([3], self.coord_i, 60, (0,1) ,r'$z_3$', r'$z_3$',False),
			'z4' : ([4], self.coord_i, 60, (0,1) ,r'$z_4$', r'$z_4$',False),
			'z5' : ([5], self.coord_i, 60, (0,1) ,r'$z_5$', r'$z_5$',False),
			'z6' : ([6], self.coord_i, 60, (0,1) ,r'$z_6$', r'$z_6$',False),
			'z7' : ([7], self.coord_i, 60, (0,1) ,r'$z_7$', r'$z_7$',False),
			'z8' : ([8], self.coord_i, 60, (0,1) ,r'$z_8$', r'$z_8$',False),
			'z9' : ([9], self.coord_i, 60, (0,1) ,r'$z_9$', r'$z_9$',False),
			#'z10' : ([0], self.coord_10, 60, (-4,4) ,r'$z_{10}$', r'z_{10}',False),
			#'z11' : ([0], self.coord_11, 60, (-4,4) ,r'$z_{11}$', r'z_{11}',False),
		}	 

		args2 = {			 
			'z0' : ([0], self.coord_0, 200, (0,1) ,r'$z_0$', r'$z_0$',False),
			'z1' : ([1], self.coord_1, 200, (0,1) ,r'$z_1$', r'$z_1$',False),
			'z2' : ([2], self.coord_2, 200, (0,1) ,r'$z_2$', r'$z_2$',False),
			'z3' : ([3], self.coord_i, 200, (0,1) ,r'$z_3$', r'$z_3$',False),
			'z4' : ([4], self.coord_i, 200, (0,1) ,r'$z_4$', r'$z_4$',False),
			'z5' : ([5], self.coord_i, 120, (0,1) ,r'$z_5$', r'$z_5$',False),
			'z6' : ([6], self.coord_i, 120, (0,1) ,r'$z_6$', r'$z_6$',False),
			'z7' : ([0], self.coord_i, 120, (0,1) ,r'$z_7$', r'$z_7$',False),
			'z8' : ([0], self.coord_i, 120, (0,1) ,r'$z_8$', r'$z_8$',False),
			'z9' : ([0], self.coord_i, 120, (0,1) ,r'$z_9$', r'$z_9$',False),
			#'z10' : ([0], self.coord_10, 120, (-4,4) ,r'$z_{10}$', r'z_{10}',False),
			#'z11' : ([0], self.coord_11, 120, (-4,4) ,r'$z_{11}$', r'z_{11}',False),
		}	 
	 
		self.args = args
		self.args2 = args2
