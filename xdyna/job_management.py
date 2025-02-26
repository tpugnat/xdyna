import xpart as xp
import numpy as np
import pandas as pd

from .da import DA
from pathlib import Path
from xaux import ProtectFile
from .da_meta import _db_access_wait_time, _db_max_lock_time






# NOT allowed on parallel process!
def jobs_status(dastudy:DA, platform:str='htcondor', **kwarg):
    # Load the Job_Manager
    from xaux import JobManager
    jm = None
    if platform == 'htcondor':
        if dastudy.meta.da_htcondor_meta.exists():
            jm = JobManager(dastudy.meta.da_htcondor_meta)
        else:
            print(f"No JobManager found for {dastudy.meta.name}.")
    elif platform == 'boinc':
        raise NotImplementedError("BOINC not yet implemented.")
    else:
        raise ValueError(f"Platform '{platform}' not supported.")
    if jm is not None:
        jm.status(platform=platform, **kwarg)


# NOT allowed on parallel process!
def jobs_retrive(dastudy:DA, platform: str='htcondor', co_search_at: str|None='ip7', **kwarg):
    # Load the Job_Manager
    from xaux import JobManager
    jm = None
    if platform == 'htcondor':
        if not dastudy.meta.da_htcondor_meta.exists():
            return
        jm = JobManager(dastudy.meta.da_htcondor_meta)
    elif platform == 'boinc':
        raise NotImplementedError("BOINC not yet implemented.")
    else:
        raise ValueError(f"Platform '{platform}' not supported.")
    jm.status(platform=platform, verbose=False)
    if jm is not None:
        results = jm.retrieve(platform=platform)
        # Load surv if not already loaded
        if dastudy._surv is None:
            dastudy.read_surv()
        if dastudy._surv is None:
            raise ValueError("No survival data found!")
        # Update surv with results
        part = xp.Particles()
        # Load line
        if dastudy.line is None:
            dastudy.load_line_from_file()
        if dastudy.line is None:
            raise ValueError("No line loaded!")
        # Update surv with results
        if dastudy.meta.nseeds != 0:
            for kk,vv in results.items():
                seed = jm._job_list[kk][0]['parameters']['seed']
                # Sometimes the CO is hard to find. Changing the starting point can help and it does not change anything for the tracking!
                if co_search_at is not None:
                    dastudy.line[seed].twiss_default['co_search_at'] = co_search_at
                # Build tracker(s) if not yet done
                if dastudy.line[seed].tracker is None:
                    print(f"Building tracker for seed {seed}.")
                    dastudy.line[seed].build_tracker()
                # Get seed and context
                context = dastudy.line[seed].tracker._buffer.context
                # Load tracking results
                with ProtectFile(vv['output_file'][0], 'rb', wait=_db_access_wait_time,
                                max_lock_time=_db_max_lock_time) as pf:
                    part = xp.Particles.from_pandas(pd.read_parquet(pf, engine="pyarrow"), _context=context)
                # Store tracking results
                part_id   = context.nparray_from_context_array(part.particle_id)
                sort      = np.argsort(part_id)
                part_id   = part_id[sort]
                x_out     = context.nparray_from_context_array(part.x)[sort]
                y_out     = context.nparray_from_context_array(part.y)[sort]
                survturns = context.nparray_from_context_array(part.at_turn)[sort]
                px_out    = context.nparray_from_context_array(part.px)[sort]
                py_out    = context.nparray_from_context_array(part.py)[sort]
                zeta_out  = context.nparray_from_context_array(part.zeta)[sort]
                delta_out = context.nparray_from_context_array(part.delta)[sort]
                s_out     = context.nparray_from_context_array(part.s)[sort]
                state     = context.nparray_from_context_array(part.state)[sort]

                dastudy._surv.loc[part_id, 'finished'] = True
                dastudy._surv.loc[part_id, 'x_out'] = x_out
                dastudy._surv.loc[part_id, 'y_out'] = y_out
                dastudy._surv.loc[part_id, 'nturns'] = survturns.astype(np.int64)
                dastudy._surv.loc[part_id, 'px_out'] = px_out
                dastudy._surv.loc[part_id, 'py_out'] = py_out
                dastudy._surv.loc[part_id, 'zeta_out'] = zeta_out
                dastudy._surv.loc[part_id, 'delta_out'] = delta_out
                dastudy._surv.loc[part_id, 's_out'] = s_out
                dastudy._surv.loc[part_id, 'state'] = state

            # Clean directory
            for vv in results[kk]['output_file']:
                vv_all_files = Path(vv).parent.glob('*')
                for vvv in vv_all_files:
                    vvv.unlink()

        else:
            for kk,vv in results.items():
                # Sometimes the CO is hard to find. Changing the starting point can help and it does not change anything for the tracking!
                if co_search_at is not None:
                    dastudy.line.twiss_default['co_search_at'] = co_search_at
                # Build tracker(s) if not yet done
                if dastudy.line.tracker is None:
                    print(f"Building tracker for seed {seed}.")
                    dastudy.line.build_tracker()
                # Get seed and context
                context = dastudy.line.tracker._buffer.context
                # Load tracking results
                with ProtectFile(vv['output_file']['0'], 'rb', wait=_db_access_wait_time,
                                max_lock_time=_db_max_lock_time) as pf:
                    part = xp.Particles.from_pandas(pd.read_parquet(pf, engine="pyarrow"), _context=dastudy._context)
                # Store tracking results
                part_id   = context.nparray_from_context_array(part.particle_id)
                sort      = np.argsort(part_id)
                part_id   = part_id[sort]
                x_out     = context.nparray_from_context_array(part.x)[sort]
                y_out     = context.nparray_from_context_array(part.y)[sort]
                survturns = context.nparray_from_context_array(part.at_turn)[sort]
                px_out    = context.nparray_from_context_array(part.px)[sort]
                py_out    = context.nparray_from_context_array(part.py)[sort]
                zeta_out  = context.nparray_from_context_array(part.zeta)[sort]
                delta_out = context.nparray_from_context_array(part.delta)[sort]
                s_out     = context.nparray_from_context_array(part.s)[sort]
                state     = context.nparray_from_context_array(part.state)[sort]

                dastudy._surv.loc[part_id, 'finished'] = True
                dastudy._surv.loc[part_id, 'x_out'] = x_out
                dastudy._surv.loc[part_id, 'y_out'] = y_out
                dastudy._surv.loc[part_id, 'nturns'] = survturns.astype(np.int64)
                dastudy._surv.loc[part_id, 'px_out'] = px_out
                dastudy._surv.loc[part_id, 'py_out'] = py_out
                dastudy._surv.loc[part_id, 'zeta_out'] = zeta_out
                dastudy._surv.loc[part_id, 'delta_out'] = delta_out
                dastudy._surv.loc[part_id, 's_out'] = s_out
                dastudy._surv.loc[part_id, 'state'] = state

            # Clean directory
            for vv in results[kk]['output_file']:
                vv_all_files = Path(vv).parent.glob('*')
                for vvv in vv_all_files:
                    vvv.unlink()
        dastudy.write_surv()

# NOT allowed on parallel process!
def jobs_resubmit(dastudy:DA, platform:str='htcondor', npart: int|None=None, njobs: int|None=None, co_search_at: str|None='ip7', **kwarg):
    # Load line
    if dastudy.line is None:
        dastudy.load_line_from_file()
    if dastudy.line is None:
        raise ValueError("No line loaded!")
    # Retrive results if JobManager already exist
    # dastudy.retrive_jobs(platform)
    # dastudy.resubmit_unfinished()
    if dastudy.meta.da_htcondor_meta.exists():
        # Load JobManager meta file
        from xaux import JobManager, DAJob
        jm = JobManager(dastudy.meta.da_htcondor_meta)
        jm.job_class = DAJob
        jm.save_metadata()
    else:
        # Set JobManager environment path
        input_directory  = dastudy.meta.path
        output_directory = Path(kwarg.pop('output_directory', dastudy.meta.path / 'htcondor' / 'output'))
        if not output_directory.exists():
            output_directory.mkdir(parents=True)
        # Generate JobManager and import DA jobs routine
        from xaux import JobManager, DAJob
        print(f"Creating JobManager for {dastudy.meta.name} in {dastudy.meta.da_htcondor_dir}.")
        jm = JobManager(name=dastudy.meta.name, work_directory=dastudy.meta.da_htcondor_dir, job_class=DAJob,
                        input_directory=input_directory, output_directory=output_directory)
    # Define the function for particle generation:
    def set_particles_per_seed(context, line, x_norm, y_norm, px_norm, py_norm, zeta, delta, nemitt_x, nemitt_y, particle_id):
        # # openmp context and radiation do not play nicely together, so temp. switch to single thread context
        # if line._context.openmp_enabled:
        #     line.discard_tracker()
        #     line.build_tracker(_context=xo.ContextCpu())
        # Create initial particles
        part = line.build_particles(
                                    x_norm=x_norm, y_norm=y_norm, px_norm=px_norm, py_norm=py_norm, zeta=zeta, delta=delta,
                                    nemitt_x=nemitt_x, nemitt_y=nemitt_y
                                    )
        part.particle_id = particle_id
        part.parent_particle_id = particle_id
        return part
    # Prepare particles submission
    if npart is None and njobs is None:
        raise ValueError("Need to specify only one of 'npart' or 'njobs'.")
    if npart is not None and njobs is not None:
        raise ValueError("Cannot specify both 'npart' and 'njobs'.")
    if npart is None:
        npart = int(np.ceil(sum(~dastudy._surv.submitted) / njobs))
    if njobs is None:
        # njobs = int(np.ceil(len(dastudy._surv[~dastudy._surv.submitted]) / npart))
        if dastudy.meta.nseeds != 0:
            njobs = int(sum([np.ceil(sum( (~dastudy._surv.finished) &  (dastudy._surv.seed==ss) ) / npart) for ss in range(1,dastudy.meta.nseeds+1)]))
        else:
            njobs = int(np.ceil(sum( (~dastudy._surv.finished) ) / npart))
# # <<<<<<<<<<<<<<<<<<<<< DEBUG
#         print(f'npart: {npart}')
#         print(f'njobs: {njobs}')
#         print(f'len(dastudy._surv[dastudy._surv.submitted]): {len(dastudy._surv[~dastudy._surv.submitted])}')
# # >>>>>>>>>>>>>>>>>>>>> DEBUG
    # select_particles = {}
    job_description = {}
    if dastudy.meta.nseeds != 0:
        for seed in range(1, dastudy.meta.nseeds+1):
            #  Select particules for the job
            mask = (dastudy._surv.submitted == False) & (dastudy._surv.seed == seed)
            if mask.sum() == 0:
                continue
            all_part_ids_seed = dastudy._surv[mask].index.to_numpy()
            # Build tracker(s) if not yet done
            # Sometimes the CO is hard to find. Changing the starting point can help and it does not change anything for the tracking!
            if co_search_at is not None:
                dastudy.line[seed].twiss_default['co_search_at'] = co_search_at
            if dastudy.line[seed].tracker is None:
                print(f"Building tracker for seed {seed}.")
                dastudy.line[seed].build_tracker()
            # Create the job
            for ii in range(0, int(np.ceil(len(all_part_ids_seed) / npart))):
                part_ids = all_part_ids_seed[ii*npart:(ii+1)*npart]
                if len(part_ids) != 0:
                    # Select initial particles
                    context = dastudy.line[seed].tracker._buffer.context
                    x_norm  = dastudy._surv.loc[part_ids, 'x_norm_in'].to_numpy()
                    y_norm  = dastudy._surv.loc[part_ids, 'y_norm_in'].to_numpy()
                    px_norm = dastudy._surv.loc[part_ids, 'px_norm_in'].to_numpy()
                    py_norm = dastudy._surv.loc[part_ids, 'py_norm_in'].to_numpy()
                    zeta    = dastudy._surv.loc[part_ids, 'zeta_in'].to_numpy()
                    delta   = dastudy._surv.loc[part_ids, 'delta_in'].to_numpy()
                    # Generate particles
                    part = set_particles_per_seed(context, dastudy.line[seed],
                                                x_norm, y_norm, px_norm, py_norm, zeta, delta,
                                                dastudy.nemitt_x, dastudy.nemitt_y, part_ids)
                    job_description[f'seed{seed}-{ii}'] = {
                        'inputfiles':{'line':dastudy.meta.line_file, 'particles':part},
                        'parameters':{'num_turns':dastudy.meta.max_turns, 'seed':seed},
                        'outputfiles':{'output_file':f'final_particles.parquet'}
                    }
        #             select_particles[f'seed{seed}-{ii}'] = [seed,part]
        # for kk,vv in select_particles.items():
        #     job_description[kk] = {'inputfiles':{'line':dastudy.meta.line_file},
        #                            'particles':vv[1],
        #                            'parameters':{'num_turns':dastudy.meta.max_turns, 'seed':vv[0]},
        #                            'outputfiles':{f'output_file':f'final_particles.parquet'}}
    else:
        #  Select particules for the job
        mask = (dastudy._surv.submitted == False)
        if mask.sum() == 0:
            return
        all_part_ids_seed = dastudy._surv[mask].index.to_numpy()
        # Build tracker(s) if not yet done
        # Sometimes the CO is hard to find. Changing the starting point can help and it does not change anything for the tracking!
        if co_search_at is not None:
            dastudy.line.twiss_default['co_search_at'] = co_search_at
        if dastudy.line.tracker is None:
            print(f"Building tracker.")
            dastudy.line.build_tracker()
        # Create the job
        for ii in range(0, np.ceil(len(all_part_ids_seed) / npart)):
            part_ids = all_part_ids_seed[ii*npart:(ii+1)*npart]
            if len(part_ids) != 0:
                # Select initial particles
                context = dastudy.line.tracker._buffer.context
                x_norm  = dastudy._surv.loc[part_ids, 'x_norm_in'].to_numpy()
                y_norm  = dastudy._surv.loc[part_ids, 'y_norm_in'].to_numpy()
                px_norm = dastudy._surv.loc[part_ids, 'px_norm_in'].to_numpy()
                py_norm = dastudy._surv.loc[part_ids, 'py_norm_in'].to_numpy()
                zeta    = dastudy._surv.loc[part_ids, 'zeta_in'].to_numpy()
                delta   = dastudy._surv.loc[part_ids, 'delta_in'].to_numpy()
                # Generate particles
                part = set_particles_per_seed(context, dastudy.line,
                                                x_norm, y_norm, px_norm, py_norm, zeta, delta,
                                                dastudy.nemitt_x, dastudy.nemitt_y, part_ids)
                job_description[f'seed{seed}-{ii}'] = {
                    'inputfiles':{'line':dastudy.meta.line_file},
                    'particles':part,
                    'parameters':{'num_turns':dastudy.meta.max_turns},
                    'outputfiles':{f'output_file':f'final_particles.parquet'}
                }
                # select_particles[f'{ii}'] = part
        # job_description = {}
        # for kk,vv in select_particles.items():
        #     job_description[kk] = {'inputfiles':{'line':dastudy.meta.line_file},
        #                            'particles':vv[1],
        #                            'parameter':{'num_turns':dastudy.meta.max_turns},
        #                            'outputfiles':{f'output_file':f'final_particles.parquet'} }
    
    if co_search_at is not None:
        for kk in job_description.keys():
            job_description[kk]['parameters']['co_search_at'] = co_search_at
    jm.add(**job_description)
    jm.submit(platform=platform, **kwarg)

# NOT allowed on parallel process!
def jobs_clean(dastudy:DA, platform:str='htcondor', **kwarg):
    # Load the Job_Manager
    from xaux import JobManager
    jm = None
    if platform == 'htcondor':
        if dastudy.meta.da_htcondor_meta.exists():
            jm = JobManager(dastudy.meta.da_htcondor_meta)
    elif platform == 'boinc':
        raise NotImplementedError("BOINC not yet implemented.")
    else:
        raise ValueError(f"Platform '{platform}' not supported.")
    jm.clean(platform=platform, **kwarg)