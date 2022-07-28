=======================
Enhanced Latent Spaces
=======================

This repo contains the code for the **E**\ nhanced **L**\ atent **S**\ p\ **A**\ ces (ELSA) framework
for neural network improved collider simulations.

Installation
-------------

Dependencies
~~~~~~~~~~~~

+---------------------------+-------------------------------+
| **Package**               | **Version**                   |
+---------------------------+-------------------------------+
| Python                    | >= 3.7                        |
+---------------------------+-------------------------------+
| Torch                     | >= 1.8                        |
+---------------------------+-------------------------------+
| Numpy                     | >= 1.20.0                     |
+---------------------------+-------------------------------+

Download + Install
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

   # clone the repository
   git clone https://github.com/ramonpeter/ELSA.git
   # then install in dev mode
   cd ELSA
   python setup.py develop


Prepare datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to download the datasets

.. code:: sh

   ./get_datasets.sh
   
This prepares and/or downloads the datasets into the **datasets** folder.
