
# Creation of the DA study environment
python -m xaux.run_da DASTUDYNAME_B1 . -c -default

# Generate the optics from madx
python -m xaux.run_da DASTUDYNAME_B1 . -l --madx_file DASTUDYNAME_B1.mask --line_file DASTUDYNAME_B1.line -sequence lhcb1 -%XING 250

# Generate the particles grid
python -m xaux.run_da DASTUDYNAME_B1 . -gp radial -default -r_min 6 -r_max 26

# Run the DA simulation on HTCondor
python -m xaux.run_da DASTUDYNAME_B1 . -rp -sb hcondor

# Check the status of the jobs
python -m xaux.run_da DASTUDYNAME_B1 . -st

# Once all jobs are done and you have all the seeds, you can run the analysis following the script:
# xdyna/examples/example_job_submission/Step4__calculate_DA.sh