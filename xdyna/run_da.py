import sys
import collections
from pathlib import Path
from pprint  import pprint
from typing  import Dict, List, Tuple

from xdyna.da import DA
# from .da_meta import _DAMetaData



USAGE = (f"Usage: {sys.argv[0]} "
         "[--help] | study [path] [-p <path>] <operand> ...")


MAIN_OPERANDS = ["-p", "--path", "-c", "--create", "-s", "--status", "-gp", "--generate_particles", 
                 "-rp", "--rerun_particles", "-t", "--track", "-htc", "--htcondor", "-l", "--line_settings",
                 "--generate_config_htcondor", "--run_config_htcondor", "--use_files", "--read_only"]   



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
        return element


def parse(args: List[str]) -> Tuple[str, Path, Dict]:
    arguments = collections.deque(args)
    DA_config = {
        'study': arguments.popleft(),
        'default_path': Path('./'),
        'use_files': False,
        'read_only': False,
        'normalised_emittance': None,
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

                if arg == '-emitt':
                    emitt = float(arguments.popleft())
                    if arguments and arguments[0][0] != '-':
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
                    raise SystemExit(f'Wrong key format starting from:\n    {arg+" ".join(arguments)}\n\n' + USAGE)
            continue

        if arg in ("-l", "--line_settings"):
            operands['Generate_line'] = {'build_line_from_madx':False, 'other_madx_flag':{}}

            while arguments:
                arg = arguments.popleft()
                if arg == "-build_from_madx":
                    operands['Generate_line']['build_line_from_madx'] = True
                    continue

                elif arg in ("-m", "--madx_file"):
                    operands['Generate_line']['madx_file'] = arguments.popleft()
                    continue

                elif arg in ("-l", "--line_file"):
                    operands['Generate_line']['line_file'] = arguments.popleft()
                    continue

                elif arg == "-other_madx_flag":
                    while arguments:
                        flag = arguments.popleft()
                        if flag[0] == '-':
                            arguments.appendleft(flag)
                            break
                        value = arguments.popleft()
                        operands['Generate_line']['other_madx_flag'][flag] = value
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
                    raise SystemExit(f'Wrong key format starting from:\n    {arg+" ".join(arguments)}\n\n' + USAGE)

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
            operands['Generate_particles'] = {'type': arguments.popleft()}

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
                    operands['Generate_particles'][arg[1:]] = converte_str_to_int_and_float(arguments.popleft())
                    arg = None
                    continue

                else:
                    raise SystemExit(f'Wrong key format starting from:\n    {arg+" ".join(arguments)}\n\n' + USAGE)
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
                    raise SystemExit(f'Wrong key format starting from:\n    {arg+" ".join(arguments)}\n\n' + USAGE)

            if arg is not None:
                raise SystemExit(f'Wrong key format starting from:\n    {arg+" ".join(arguments)}\n\n' + USAGE)
        if arg is not None and arg not in MAIN_OPERANDS:
            raise SystemExit(f'Wrong key format starting from:\n    {arg+" ".join(arguments)}\n\n' + USAGE)
    

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
    default_path = config.pop(['default_path'])
    if 'Create' not in operands:

        if (default_path / config['study']+'meta.json').is_file() or \
            (default_path / config['study']+'meta.csv').is_file():
            path = default_path

        elif (default_path / config['study'] / config['study']+'meta.json').is_file() or \
            (default_path / config['study'] / config['study']+'meta.csv').is_file():
            path = default_path / config['study']

        else:
            raise FileNotFoundError(f"{config['study']} not found in {default_path}")
        
        if (path / config['study']+'meta.json').is_file():
            return DA(path=path, **config)
        
        elif (path / config['study']+'meta.csv').is_file():
            from xdyna.da import load_sixdesk_output
            return load_sixdesk_output(path=path, study=config['study'], nemit = config['normalised_emittance'])
        
        else:
            raise FileNotFoundError(f"Study {config['study']} not found in {path}")
        
    else:
        if not (default_path / config['study']+'meta.json').is_file():
            return DA(path=default_path, **config, **operands['Create'])
        else:
            raise FileExistsError(f"Study {config['study']} already exists in {default_path}")
        

def get_line(da_study: DA, config:dict):
    if 'madx_file' in config:
        da_study.madx_file = config.pop('madx_file')
    if 'line_file' in config:
        if da_study.line_file is None:
            da_study.line_file = config.pop('line_file')
        elif da_study.line_file != config['line_file'] and config['build_line_from_madx']:
            raise ValueError(f"Line file already exists: {da_study.line_file} and {config['line_file']}")
        else:
            da_study.line_file = config.pop('line_file')


    if not da_study.line_file.exist() and config['build_line_from_madx']:
        da_study.build_line_from_madx(**config)
    else:
        da_study.load_line_from_file()


def generate_particles(da_study: DA, config:dict):
    type_dist = config.pop('type')
    if type_dist == 'random':
        da_study.generate_random_initial(**config)
    elif type_dist == 'radial':
        da_study.generate_initial_radial(**config) 
    elif type_dist == 'grid':
        da_study.generate_initial_grid(**config)
    else:
        raise ValueError(f"Type of distribution {type_dist} not implemented. `random`, `radial` or `grid` are allowed")


def generate_config_htcondor(da_study: DA):
    import numpy as np
    list_npart = [200, 500, 1000, 2000, 5000]
    list_jobflavour = ["workday", "tomorrow", "tomorrow", "testmatch", "nextweek"]

    # npart = 200
    # jobflavour = "tomorrow"
    njobs=-1
    if da_study.meta.nseeds != 0:
        for npt, jbflvr in zip(list_npart, list_jobflavour):
            if njobs == -1 or njobs > 500:
                npart = npt
                jobflavour = jbflvr
                njobs = sum([np.ceil(sum( (~DA._surv.finished) &  (DA._surv.seed==ss) ) / npart) for ss in range(1,da_study.meta.nseeds+1)])
    else:
        for npt, jbflvr in zip(list_npart, list_jobflavour):
            if njobs == -1 or njobs > 500:
                npart = npt
                jobflavour = jbflvr
                njobs = np.ceil(sum( ~DA._surv.finished ) / npart)

    path = da_study.meta.path
    study = da_study.meta.name
    txt = ''
    txt+= '[DEFAULT]\n'
    txt+= 'executable="bash"\n'
    # TODO: Add the path to the mask
    txt+= 'mask="/afs/cern.ch/work/t/thpugnat/public/DA_study_with_xdyna/mask_track.sh"\n'
    # TODO: Add the path to the working directory
    txt+= f'working_directory={"/afs/cern.ch/work/t/thpugnat/private/xtrack_tests/DA/hcondor/local/"+study}\n'
    txt+= '# "espresso", "microcentury", "longlunch", "workday", "tomorrow", "testmatch", "nextweek"\n'
    txt+= f'jobflavour="{jobflavour}"\n'
    txt+= 'run_local=False\n'
    txt+= 'resume_jobs=False\n'
    txt+= 'append_jobs=False\n'
    txt+= 'dryrun=False\n'
    txt+= 'num_processes=8\n'

    txt+= 'htc_arguments={"accounting_group":"group_u_BE.ABP.normal","MY.WantOS":"el9"}\n'
    ljobs=str([ii for ii in range(njobs)])
    txt+= "\n\nreplace_dict={ "
    txt+= f"'ff':['{study}'], "
    txt+= f"'npart': [{npart}], 'i': {ljobs}" + "}\n"

    file = Path(path / f"config_htcondor_track_{study}.ini")
    if file.exists():
        file.unlink()

    with open(file,'w') as pf:
        pf.write(txt)

    print(f"python -m pylhc_submitter.job_submitter --entry_cfg {file}")

    # raise NotImplementedError("Status not implemented yet")


def run_htcondor(da_study: DA):
    file = Path(da_study.meta.path / f"config_htcondor_track_{da_study.meta.name}.ini")
    if file.exists():
        if 'pylhc_submitter' in sys.modules:
            import os
            os.system(f"python -m pylhc_submitter.job_submitter --entry_cfg {file}")
        else:
            raise ImportError("Module pylhc_submitter not found")
    else:
        raise FileNotFoundError(f"File {file} not found")

    raise NotImplementedError("Status not implemented yet")


def track(da_study: DA, config:dict):
    da_study.track_job(**config)


def status(da_study: DA):
    # da_study.status()
    raise NotImplementedError("Status not implemented yet")
# =================================================================================================







# =================================================================================================
def main() -> None:
    config, operands = parse(sys.argv[1:])
    if not operands:
        raise SystemExit(USAGE)
    print(f"config:")
    pprint(config, indent=4)
    print(f"operands:")
    pprint(operands, indent=4)

    # da_study = get_DA(config, operands)
    # if 'Generate_line' in operands:
    #     get_line(da_study, operands['Generate_line'])

    # if  'Generate_particles' in operands:
    #     generate_particles(da_study, operands['Generate_particles'])

    # if 'Rerun_particles' in operands:
    #     da_study.resubmit_unfinished()

    # if 'Status' in operands:
    #     status(da_study)

    # if 'Generate_config_htcondor' in operands:
    #     generate_config_htcondor(da_study)

    # if 'Run_config_htcondor' in operands:
    #     run_htcondor(da_study)  

    # if 'Track' in operands:
    #     track(da_study, operands['Track'])
# =================================================================================================





if __name__ == "__main__":
    main()