# Supplementary material for the article
# *Dynamic resource allocation for controlling pathogen spread on a large animal metapopulation network* (Cristancho-Fajardo L., Ezanno P., Vergu E.).
Python Simulation Code and simulated datasets  for the article Dynamic resource allocation for controlling pathogen spread on a large animal metapopulation network* (Cristancho-Fajardo L., Ezanno P., Vergu E.):

- The fixed_parameters folder contains the fixed simulation setting, fixed network structure and fixed demographic parameters. As well as the notebook on their generation. 

- The vaccination folder, contains:

  - The dynamic folder, contains:
        - The score_list folder: contains the text files with the names of the score functions to be tested. As well as the notebook on their generation. 
        - The vaccination_functions.py script gathers the main functions used in the generation of the simulated epidemic under vaccination allocation
        - launcher_vacc_endemicdynamic.py is the script that generates the simulated dynamics under vaccination, in the endemic scenario.
        - launcher_vacc_epidemicdynamic.py is the script that generates the simulated dynamics under vaccination, in the epidemic scenario.
        - simulated_data_vaccend_dynamic is the folder that contains the simulated datasets for vaccination with b_fix = 25\% of initial total population, in the           endemic scenario.
        - simulated_data_vaccepi_dynamic is the folder that contains the simulated datasets for vaccination with b_fix = 25\% of initial total population, in the           epidemic scenario.
  - The percolation folder, contains:
        - The score_budget_list folder: contains the text files with the combinations of the score functions and quantity of resource (b_fix) to be tested. 
          As well as the notebook on their generation. 
        - The vaccination_functions.py script gathers the main functions used in the generation of the simulated epidemic under vaccination allocation
        - launcher_percolation_vaccendemic.py is the script that generates the percolation experiments for vaccination, in the endemic scenario.
        - launcher_percolation_vaccepidemic.py is the script that generates the percolation experiments for vaccination, in the epidemic scenario.
        - percolation_vacc_endemic_data is the folder that contains the simulated datasets in the percolation experiments for vaccination,
          in the endemic scenario.
        - percolation_vacc_epidemic_data is the folder that contains the simulated datasets in the percolation experiments for vaccination,
          in the epidemic scenario.
          
  - The plots_vaccination.ipynb is a jupyter-notebook that reads the simulated datasets and generates the figures presented in the article that concern
    vaccination.


- The treatment folder, contains:

  - The dynamic folder, contains:
        - The score_list folder: contains the text files with the names of the score functions to be tested. As well as the notebook on their generation. 
        - The treatment_functions.py script gathers the main functions used in the generation of the simulated epidemic under treatment allocation
        - launcher_treatment_endemic_dynamic.py is the script that generates the simulated dynamics under treatment, in the endemic scenario.
        - launcher_treatment_epidemic_dynamic.py is the script that generates the simulated dynamics under treatment, in the epidemic scenario.
        - simulated_data_treatend_dynamic is the folder that contains the simulated datasets for treatment with b_fix = 25 herds, in the endemic scenario.
        - simulated_data_treatepi_dynamic is the folder that contains the simulated datasets for treatment with b_fix = 25 herds, in the epidemic scenario.
  - The percolation folder, contains:
        - The score_budget_list folder: contains the text files with the combinations of the score functions and quantity of resource (b_fix) to be tested. 
          As well as the notebook on their generation. 
        - The treatment_functions.py script gathers the main functions used in the generation of the simulated epidemic under treatment allocation
        - launcher_percolation_treatment_endemic.py is the script that generates the percolation experiments for treatment, in the endemic scenario.
        - launcher_percolation_treatment_epidemic.py is the script that generates the percolation experiments for treatment, in the epidemic scenario.
        - percolation_treatment_endemic_data is the folder that contains the simulated datasets in the percolation experiments for treatment,
          in the endemic scenario.
        - percolation_treatment_epidemic_data is the folder that contains the simulated datasets in the percolation experiments for treatment,
          in the epidemic scenario.
          
  - The plots_treatment.ipynb is a jupyter-notebook that reads the simulated datasets and generates the figures presented in the article that concern
    treatment.

Copyright or © or Copr. [INRAE]

Contributor(s) : [Lina Cristancho-Fajardo]  ([2021])

[lina.cristancho-fajardo@inrae.fr]

This software is a computer program whose purpose is to simulate the stochastic spread of a pathogen on an animal trade-network, by generating social planner's dynamic resource allocation decisions (on vaccination or a treatment) on the pathogen's spread according to a certain indicator or scoring function for determining herds priority. 

This software is governed by the [CeCILL-B] license under French law and
abiding by the rules of distribution of free software.  You can  use, 
modify and/ or redistribute the software under the terms of the [CeCILL-B]
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info". 

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability. 

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or 
data to be ensured and,  more generally, to use and operate it in the 
same conditions as regards security. 

*The fact that you are presently reading this means that you have had
knowledge of the [CeCILL-B] license and that you accept its terms.*
