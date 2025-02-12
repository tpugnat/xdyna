import sys
import collections
from pathlib import Path
from pprint  import pprint
from typing  import Dict, List, Tuple

print("PYTHONPATH:", sys.path)

# import xtrack
# import xpart
from .da import DA
# from xdyna.da import DA
# from .da_meta import _DAMetaData
# DA = int


USAGE = (f"Usage: {sys.argv[0]} "
         "[--help] | study [path] [-p <path>] <operand> ...")


MAIN_OPERANDS = ["-p", "--path", "-c", "--create", "-s", "--status", "-gp", "--generate_particles", 
                 "-rp", "--rerun_particles", "-t", "--track", "-htc", "--htcondor", "-l", "--line_settings",
                 "--generate_config_htcondor", "--run_config_htcondor", "--use_files", "--read_only"]   

DEFAULT_GENERATE_PARTICLES_PARAMETERS = {
    'radial': {'r_min': 6, 'r_max': 20, 'delta':0.00027, 'pairs_shift':1e-7, 'pairs_shift_var':'x'},
    'grid': {},
    'random': {},
}

DEFAULT_CREATE_PARAMETERS = {'nseeds':60, 'max_turns':1e5, 'emitt': 2.5e-6}

# =================================================================================================
def converte_str_to_int_and_float(element: any):
    #If you expect None to be passed:
    if element is None: 
        return element
    try:
        if '.' in element or 'e' in element.lower():
            nelement = float(element)
        else:
            nelement = int(element)
        return nelement
    except ValueError:
        if element.lower() == 'true' or element.lower() == 't':
            return True
        elif element.lower() == 'false' or element.lower() == 'f':
            return False
        else:
            return element


def parse(args: List[str]) -> Tuple[str, Path, Dict]:
    arguments = collections.deque(args)
    DA_config = {
        'study': arguments.popleft(),
        'default_path': Path('./'),
        'use_files': False,
        'read_only': False,
        # 'normalised_emittance': None,
    }
    operands: Dict = {}

    if DA_config['study'] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0)
    
    if arguments[0][0] != '-':
        DA_config['default_path'] = Path(arguments.popleft())
    
    while arguments:
        arg = arguments.popleft()
        # Default to operands
        if arg in ("-h", "--help"):
            print(USAGE)
            sys.exit(0)

        if arg in ("-p", "--path"):
            DA_config['default_path'] = Path(arguments.popleft())
            continue

        if arg == "--use_files":
            DA_config['use_files'] = True
            if arguments and arguments[0][0] != '-':
                DA_config['use_files'] = bool(arguments.popleft())
            continue

        if arg == '-emitt':
            emitt = float(arguments.popleft())
            if (len(arguments) != 0) and (arguments[0][0] != '-'):
                emitt = (emitt,float(arguments.popleft()))
            DA_config['normalised_emittance'] = emitt
            continue

        if arg == "--read_only":
            DA_config['read_only'] = True
            if arguments and arguments[0][0] != '-':
                DA_config['read_only'] = bool(arguments.popleft())
            continue

        # Specific to DA commands
        if arg in ("-c", "--create"):
            operands['Create'] = {'max_turns': int(1e5),}

            while arguments:
                arg = arguments.popleft()

                if arg == '-default':
                    diff_operands = DEFAULT_CREATE_PARAMETERS.copy()
                    emitt = diff_operands.pop('emitt')
                    if 'normalised_emittance' not in DA_config:
                        DA_config['normalised_emittance'] = emitt
                    diff_operands = {kk:vv for kk,vv in diff_operands.items() if kk not in operands['Create']}
                    operands['Create'] = {**operands['Create'], **diff_operands}

                elif arg == '-emitt':
                    emitt = float(arguments.popleft())
                    if (len(arguments) != 0) and (arguments[0][0] != '-'):
                        emitt = (emitt,float(arguments.popleft()))
                    DA_config['normalised_emittance'] = emitt
                    continue

                elif arg in MAIN_OPERANDS:
                    arguments.appendleft(arg)
                    arg = None
                    break

                elif arguments and arguments[0][0] != '-':
                    operands['Create'][arg[1:]] = converte_str_to_int_and_float(arguments.popleft())
                    arg = None
                    continue

                else:
                    raise SystemExit(f'Wrong key format starting from:\n    {arg+" "+" ".join(arguments)}\n\n' + USAGE)
            continue

        if arg in ("-l", "--line_settings"):
            # operands['Generate_line'] = {'build_line_from_madx':False, 'other_madx_flag':{}}
            operands['Generate_line'] = {'other_madx_flag':{}}

            while arguments:
                arg = arguments.popleft()
                # if arg == "-build_from_madx":
                #     operands['Generate_line']['build_line_from_madx'] = True
                #     continue

                if arg in ("-m", "--madx_file"):
                    operands['Generate_line']['madx_file'] = arguments.popleft()
                    continue

                elif arg in ("-l", "--line_file"):
                    operands['Generate_line']['line_file'] = arguments.popleft()
                    continue

                # elif arg == "-other_madx_flag":
                #     while arguments:
                #         flag = arguments.popleft()
                #         if flag[0] == '-':
                #             arguments.appendleft(flag)
                #             break
                #         value = arguments.popleft()
                #         operands['Generate_line']['other_madx_flag'][flag] = value
                #     continue

                elif arg in MAIN_OPERANDS:
                    arguments.appendleft(arg)
                    arg = None
                    break

                elif arguments and arguments[0][0] != '-':
                    operands['Create'][arg[1:]] = converte_str_to_int_and_float(arguments.popleft())
                    arg = None
                    continue

                else:
                    raise SystemExit(f'Wrong key format starting from:\n    {arg+" "+" ".join(arguments)}\n\n' + USAGE)

        # if arg in ("-m", "--madx_file"):
        #     if 'Generate_line' not in operands:
        #         operands['Generate_line'] = {}
        #     operands['Generate_line']['madx_file'] = arguments.popleft()
        #     continue

        # if arg in ("-l", "--madx_file"):
        #     if 'Generate_line' not in operands:
        #         operands['Generate_line'] = {}
        #     operands['Generate_line']['madx_file'] = arguments.popleft()
        #     continue

        if arg in ("-s", "--status"):
            operands['Status'] = True
            continue

        if arg in ("-gp", "--generate_particles"):
            particle_type = arguments.popleft()
            operands['Generate_particles'] = {'type': particle_type}

            while arguments:
                arg = arguments.popleft()

                if arg == '-default':
                    diff_operands = {kk:vv for kk,vv in DEFAULT_GENERATE_PARTICLES_PARAMETERS[particle_type].items() if kk not in operands['Generate_particles']}
                    operands['Generate_particles'] = {**operands['Generate_particles'], **diff_operands}

                elif arg in MAIN_OPERANDS:
                    arguments.appendleft(arg)
                    arg = None
                    break

                elif arguments and arguments[0][0] != '-':
                    operands['Generate_particles'][arg[1:]] = converte_str_to_int_and_float(arguments.popleft())
                    arg = None
                    continue

                else:
                    raise SystemExit(f'Wrong key format starting from:\n    {arg+" "+" ".join(arguments)}\n\n' + USAGE)
            continue

        if arg in ("-rp", "--rerun_particles"):
            operands['Rerun_particles'] = True
            continue

        if arg in ("-htc","--htcondor","--generate_config_htcondor"):
            operands['Generate_config_htcondor'] = True
            if arg not in ("-htc","--htcondor"):
                continue

        if arg in ("-htc","--htcondor","--run_config_htcondor"):
            operands['Run_config_htcondor'] = True
            continue

        if arg in ("-t", "--track"):
            operands['Track'] = {'npart':None}

            while arguments:
                arg = arguments.popleft()
                # if arg == "--use_files":
                #     DA_config['use_files'] = True
                #     if arguments and arguments[0][0] != '-':
                #         DA_config['use_files'] = bool(arguments.popleft())
                #     continue

                # elif arg == "--read_only":
                #     DA_config['read_only'] = True
                #     if arguments and arguments[0][0] != '-':
                #         DA_config['read_only'] = bool(arguments.popleft())
                #     continue

                # elif arg in ("-m", "--madx_file"):
                #     if 'Generate_line' not in operands:
                #         operands['Generate_line'] = {}
                #     operands['Generate_line']['madx_file'] = arguments.popleft()
                #     continue

                # elif arg in ("-l", "--madx_file"):
                #     if 'Generate_line' not in operands:
                #         operands['Generate_line'] = {}
                #     operands['Generate_line']['madx_file'] = arguments.popleft()
                #     continue

                if arg in MAIN_OPERANDS:
                    arguments.appendleft(arg)
                    arg = None
                    break

                elif arguments and arguments[0][0] != '-':
                    operands['Run'][arg[1:]] = converte_str_to_int_and_float(arguments.popleft())
                    arg = None
                    continue

                else:
                    raise SystemExit(f'Wrong key format starting from:\n    {arg+" "+" ".join(arguments)}\n\n' + USAGE)

            if arg is not None:
                raise SystemExit(f'Wrong key format starting from:\n    {arg+" "+" ".join(arguments)}\n\n' + USAGE)
        if arguments and arg is not None and arg not in MAIN_OPERANDS:
            raise SystemExit(f'Wrong key format starting from:\n    {arg+" "+" ".join(arguments)}\n\n' + USAGE)
    

        # try:
        #     operands.append(int(arg))
        # except ValueError:
        #     raise SystemExit(USAGE)
        # if len(operands) > 3:
        #     raise SystemExit(USAGE)

    return DA_config, operands
# =================================================================================================







# =================================================================================================
def get_DA(config: Dict, operands: Dict):
    path = Path(config.get('default_path'))
    study= config.get('study')
    if 'Create' not in operands:
        # Check if the study already exists in different path
        if (path / study / (study+'.meta.json')).exists() or \
            (path / study / (study+'.meta.csv')).exists():
            path = path / study
        elif not ((path / (study+'.meta.json')).exists() or \
            (path / (study+'.meta.csv')).exists()):
            raise FileNotFoundError(f"{study} not found in {path}")
        # Load the study
        if (path / (study+'.meta.json')).exists():
            print(f'   -> Loading study {study} from {path}')
            return DA(name=study, path=path, **config)
        elif (path / (study+'.meta.csv')).exists():
            print(f'   -> Loading study {study} from SixDesk input located at {path}')
            from xdyna.da import load_sixdesk_output
            if 'normalised_emittance' not in config:
                config['normalised_emittance'] = 2.5
            return load_sixdesk_output(path=path, study=study, nemit = config['normalised_emittance'])
        else:
            raise FileNotFoundError(f"Study {config['study']} not found in {path}")
    else:
        # Check if the study already exists in different path and return an error, else create the study
        if  (path / (study+'.meta.json')).exists():
            raise FileExistsError(f"Study {study} already exists in {path}")
        elif  (path / study / (study+'.meta.json')).exists():
            raise FileExistsError(f"Study {study} already exists in {path / study}")
        elif  (path / (study+'.meta.csv')).exists():
            raise FileExistsError(f"Study {study} already exists in {path} as SixDesk input")
        elif  (path / study / (study+'.meta.csv')).exists():
            raise FileExistsError(f"Study {study} already exists in {path / study} as SixDesk input")
        else:
            if not path.exists():
                path.mkdir(parents=True)
            print(f'   -> Create study {study} at {path}')
            return DA(name=study, path=path, **config, **operands['Create'])
        
        

def get_line(da_study: DA, config:dict):
    print(f'   -> Loading the line')
    # Manage the madx mask file
    if 'madx_file' in config:
        print(f'      -> A madx mask has been specified and copy has been made in the study directory.')
        madx_file = config.pop('madx_file')
        da_study.madx_file = da_study.meta.path / (da_study.meta.name+'.madx')
        import shutil
        shutil.copy(madx_file,da_study.madx_file)
    # Manage the line file
    if 'line_file' in config:
        if da_study.line_file is None:
            print(f'      -> A line file has been specified.')
            da_study.line_file = config.pop('line_file')
        elif da_study.line_file != config['line_file']:
            raise ValueError(f"Line file already exists: {da_study.line_file} and {config['line_file']}")
        else:
            da_study.line_file = config.pop('line_file')
    elif da_study.madx_file is not None:
        print(f'      -> A line file will be created in the study directory.')
        da_study.line_file = da_study.meta.path / (da_study.meta.name+'.line.json')
    # Generate the line if a mask has been given.
    if not da_study.line_file.exists():
        if da_study.madx_file is not None:
            print(f'      -> The line will be created with extra variables:')
            if 'sequence' in config:
                sequence = config.pop('sequence')
            else:
                bb = int(da_study.meta.name[-1])
                sequence= 'lhcb1' if bb<3 else 'lhcb2'
            print(f'         -> The selected sequence is {sequence}')
            variable_to_replace_in_mask= {'%EMIT_BEAM': 2.5,'%NPART': 1,'%XING': 250,}
            for key in config.keys():
                if key[0] == '%':
                    variable_to_replace_in_mask[key] = config.pop(key)
            print(f'         -> The following parameter will be replace in the mask: {variable_to_replace_in_mask}')
            if 'run_all_seeds' not in config:
                config['run_all_seeds'] = (da_study.meta.nseeds!=0)
            da_study.build_line_from_madx(sequence=sequence, other_madx_flag= variable_to_replace_in_mask, **config)
        else:
            raise FileNotFoundError(f"Either a madx file and/or a line file should be provided")
    # else:
    #     print(f'      -> The line will be loaded.')
    #     da_study.load_line_from_file()


def generate_particles(da_study: DA, config:dict):
    type_dist = config.pop('type')
    if type_dist == 'random':
        print(f'   -> Generating random initial distribution of particles')
        da_study.generate_random_initial(**config)
    elif type_dist == 'radial':
        print(f'   -> Generating radial initial distribution of particles')
        if 'r_num' not in config and 'r_step' not in config:
            config['r_num'] = (29*( (config['r_max']-config['r_min']) // 2 ) +1)
        da_study.generate_initial_radial(**config) 
    elif type_dist == 'grid':
        print(f'   -> Generating grid initial distribution of particles')
        da_study.generate_initial_grid(**config)
    else:
        raise ValueError(f"Type of distribution {type_dist} not implemented. `random`, `radial` or `grid` are allowed")


def generate_config_htcondor(da_study: DA):
    raise NotImplementedError("generate_config_htcondor not implemented yet")
    path = da_study.meta.path
    study = da_study.meta.name

    # Clean the config file if it exists
    file = Path(path / f"config_htcondor_track_{study}.ini")
    if file.exists():
        file.unlink()

    # Count the number of jobs to run on htcondor
    import numpy as np
    list_npart = [200, 500, 1000, 2000, 5000]
    list_jobflavour = ["workday", "tomorrow", "tomorrow", "testmatch", "nextweek"]

    # npart = 200
    # jobflavour = "tomorrow"
    njobs=-1
    if da_study.meta.nseeds != 0:
        mask = ~da_study._surv.finished
        for npt, jbflvr in zip(list_npart, list_jobflavour):
            if njobs == -1 or njobs > 500:
                npart = npt
                jobflavour = jbflvr
                njobs = int(sum([np.ceil(sum( mask & (da_study._surv.seed==ss) ) / npart) for ss in range(1,da_study.meta.nseeds+1)]))
    else:
        for npt, jbflvr in zip(list_npart, list_jobflavour):
            if njobs == -1 or njobs > 500:
                npart = npt
                jobflavour = jbflvr
                njobs = int(np.ceil(sum( ~da_study._surv.finished ) / npart))

    if njobs != 0:
        # Create the config file
        txt = ''
        txt+= '[DEFAULT]\n'
        txt+= 'executable="bash"\n'
        # TODO: Add the path to the mask
        txt+= 'mask="/afs/cern.ch/work/t/thpugnat/public/DA_study_with_xdyna/mask_track.sh"\n'
        # txt+=f'mask="{path / "hcondor/mask_track.sh"}"\n'
        # TODO: Add the path to the working directory
        txt+= f'working_directory="{path / "hcondor/"}"\n'
        txt+= '# "espresso", "microcentury", "longlunch", "workday", "tomorrow", "testmatch", "nextweek"\n'
        txt+= f'jobflavour="{jobflavour}"\n'
        txt+= 'run_local=False\n'
        txt+= 'resume_jobs=False\n'
        txt+= 'append_jobs=False\n'
        txt+= 'dryrun=False\n'
        txt+= 'num_processes=8\n'

        txt+= 'htc_arguments={"accounting_group":"group_u_BE.ABP.normal","MY.WantOS":"el9","batch_name":"' +str(study)+'"}'
        ljobs=str([ii for ii in range(njobs)])
        txt+= "\n\nreplace_dict={ "
        txt+= f"'ff':['{study}'], "
        txt+= f"'npart': [{npart}], 'i': {ljobs}" + "}\n"

        with open(file,'w') as pf:
            pf.write(txt)

        print(f"python -m pylhc_submitter.job_submitter --entry_cfg {file}")

    # raise NotImplementedError("Status not implemented yet")


def run_htcondor(da_study: DA):
    raise NotImplementedError("run_htcondor not implemented yet")
    file = Path(da_study.meta.path / f"config_htcondor_track_{da_study.meta.name}.ini")
    if file.exists():
        if 'pylhc_submitter' in sys.modules:
            import os
            os.system(f"python -m pylhc_submitter.job_submitter --entry_cfg {file}")
        else:
            raise ImportError("Module pylhc_submitter not found")
    else:
        raise FileNotFoundError(f"File {file} not found! Be sure to generate the config file first using" +\
                                " `-htc`,`--htcondor` or `--generate_config_htcondor`")

    # raise NotImplementedError("Status not implemented yet")


def track(da_study: DA, config:dict):
    da_study.track_job(**config)


def status(da_study: DA):
    raise NotImplementedError("Status not implemented yet")
    da_study.status()
# =================================================================================================







# =================================================================================================
def run_da(argv) -> None:
    config, operands = parse(argv)
    if not operands:
        raise SystemExit(USAGE)
    print(f"config:")
    pprint(config, indent=4)
    print(f"operands:")
    pprint(operands, indent=4)

    # import xdyna
    # print(Path(xdyna.__file__).parent)

    da_study = get_DA(config, operands)
    if 'Generate_line' in operands:
        get_line(da_study, operands['Generate_line'])

    if  'Generate_particles' in operands:
        generate_particles(da_study, operands['Generate_particles'])

    if 'Rerun_particles' in operands:
        da_study.resubmit_unfinished()

    # if 'Status' in operands:
    #     status(da_study)

    # if 'Generate_config_htcondor' in operands:
    #     generate_config_htcondor(da_study)

    # if 'Run_config_htcondor' in operands:
    #     run_htcondor(da_study)  

    # if 'Track' in operands:
    #     track(da_study, operands['Track'])
# =================================================================================================



if __name__ == "__main__" or __name__ == "__xdyna.run_da__":
    run_da(sys.argv[1:])