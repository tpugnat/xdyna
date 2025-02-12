# xdyna

Tools to study beam dynamics in xtrack simulations, like dynamic aperture calculations, PYTHIA integration, dynamic indicators, ...

## Dynamic aperture studies

The `xdyna` package provides the `DA` class which serves as a simple front-end for setting up and running dynamic aperture studies.

To start, a `xtrack.line` object is required.
The following code then sets up the study and launches the tracking

```python

import xdyna as xd

da = xd.DA(
    name='name_of_your_study', # used to generate a directory where files are stored
    normalised_emittance=[1,1], # provide normalized emittance for particle initialization in [m]
    max_turns=1e5, # number of turns to track
    use_files=False 
    # in case DA studies should run on HTC condor, files are used to collect the information
    # if the tracking is performed locally, no files are needed
)
    
# initialize a grid of particles using 5 angles in x-y space, in a range from 0 to 20 sigmas in steps of 5 sigma.
da.generate_initial_radial(angles=5, r_min=0, r_max=20, r_step=5, delta=0.) 

da.line = line # associate prev. created line, holding the lattice and context, with DA object

da.track_job() # start the tracking

da.survival_data # returns a dataframe with the number of survived turns for the initial position of each particle

```

To use on a platform like HTCondor, perform the same setup as before but using `use_files=True`.
Each HTCondor job then only requires the following lines

```python
import xdyna as xd
# This will load the existing DA based on the meta file with the same name found in the working directory.
# If the script is ran somewhere else, the path to the metafile can be passed with 'path=...'.
DA = xd.DA(name=study, use_files=True)

# Do the tracking, here for 100 particles.
# The code will automatically look for particles that are not-submitted yet and use these.
DA.track_job(npart=100)
```

## Command line description

With the latest update,`xdyna` can be run directly from the a Terminal. 
Those command can handle the basic functions from `xdyna` such as: study creation; line generation/loading; particles generation; job submission to htcondor/boinc; showing the study status; and tracking.

`xdyna` is call using the command `python -m xaux.run_da <study> [<path>] <operands>`. 
If `<path>` is not specified, `xdyna` will look for meta file in the current directory or in any directory with the same name as the study.

Here are examples of the different functionnalities and a quick description of the associated operand:

### - Class creation [`-c`, `--create`]:

This option is used for the creation of a new study and can be followed by the element of the class initialisation description as such:
```Shell
python -m xaux.run_da <study> [<path>] -c -nseeds 60 -max_turns 1e5 -emitt 2.5e-6
python -m xaux.run_da <study> [<path>] -c -nseeds 60 -max_turns 1e5 -emitt 2.5e-6 2.5e-6
```

A fast command for typical DA studies with 60 seeds can be called using the command `-default`. 
Any added command based on the class initialisation description will overwrite the default settings.
```Shell
python -m xaux.run_da <study> [<path>] -c -default
python -m xaux.run_da <study> [<path>] -c -default -nseeds 0
```

### - Line creation and loading [`-l`, `--line_settings`]:

This option is used for the creation and/or loading of the line(s).
It must be followed by either the command for the madx mask file and/or  the line file. 
If not, the line in memory will be loaded.
```Shell
python -m xaux.run_da <study> [<path>] -l --line_file <line_file>
python -m xaux.run_da <study> [<path>] -l --madx_file <madx_file> --line_file <line_file>
```

If any option starting with `%` will be replace in the mask. Any other option will be passed  as parameter to either `build_line_from_madx` if the line file does not exist, or `load_line_from_file`.
```Shell
python -m xaux.run_da <study> [<path>] -l --madx_file <madx_file> -sequence lhcb1 -%XING 250
```

### - Generation of the particle distribution [`-gp`, `--generate_particles`]:

This option is used for the generation of the particle distribution and must be followed by the type: `"radial"`, `"grid"`, or `"random"`.
The option after correspond to the argument of the respective distribution generator function.
```Shell
python -m xaux.run_da <study> [<path>] -gp radial -r_min 10 -r_max 20 -angles 11
```
if `r_num` and `r_step` are not specified, the 30 particles will be distributed over step of 2 sigma.

A fast command for typical DA studies with 60 seeds can be called using the command `-default`. 
Any added command based on the distribution generator description will overwrite the default settings.
```Shell
python -m xaux.run_da <study> [<path>] -gp radial -default
python -m xaux.run_da <study> [<path>] -gp radial -default -r_min 10
```

### - Submission to parallel computation [`-sb`, `--submit`]:
This option is used for the generation of the particle distribution and must be followed by the name of the platform: `"htcondor"` or `"boinc"`.
Results from finished jobs are automatically loaded when called.
It is possible to disable or enable the auto jobs submission to the platform using the command `-auto` followed by a boulean. If disable, the line for the submission will be only printed. By default, it is enable.
Similarly, it is possible to disable or enable the auto cleaning of the platform submission directory using the command `-clean` followed by a boulean. By default, it is disable.
The option after correspond to the argument of the respective distribution generator function.
```Shell
python -m xaux.run_da <study> [<path>] -sb hcondor
python -m xaux.run_da <study> [<path>] -sb hcondor -auto False -clean True
```

### - Show the status of the study [`-st`, `--status`]:
This option will print different information concerning the status of the study such as the number of seed, the number of submitted and finished particles, ongoing simulations on htcondor, ...
Any added command will be pass to `xdyna` status function.
```Shell
python -m xaux.run_da <study> [<path>] -st
```

### - Track the particles [`-t`, `--track`]:
This option will track particles, once seed at a time.
Any added command will be pass to `xdyna` track function.
```Shell
python -m xaux.run_da <study> [<path>] -t
python -m xaux.run_da <study> [<path>] -t -npart 100
```
