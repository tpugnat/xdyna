import numpy as np
import matplotlib.pyplot as plt
from .postprocess_tools import fit_DA


def plot_particles(DA, ax=None, at_turn=None, type_plot="polar", seed=None,
                   closses="red", csurviving="blue", size_scaling="log", **kwargs):
    """Scatter plot of the lost and surviving particles.

Parameters
----------
ax:           Plot axis.
at_turn:      All particles surviving at least this number of turns are considered as surviving.
seed:         In case of multiseed simulation, the seed number must be specified (Default=None).
type_plot:    x-y for cartesian, ang-amp for polar (Default="polar").
csurviving:   Color of surviving dots (Default="blue"). Use "" to disable.
closses:      Color of losses dots (Default="red"). Use "" to disable.
size_scaling: Type of losses dot scaling (Default="log"). There are 3 options: "linear", "log", None.
"""

    if ax is None:
        ax = plt.subplots(1,1,figsize=(10,10))[1]

    if DA.survival_data is None:
        raise ValueError('Run the simulation before using plot_particles.')
    if DA.meta.nseeds>0 and (seed==None or seed=='stat'):
        raise ValueError('For multiseed simulation, please specify a seed number.')

    # Clean kwargs and initiallize parameters
    if 'c' in kwargs:
        kwargs.pop('c')
        print("Warning: ignoring parameter 'c'. Use 'closses' and 'csurviving' instead.")
    if 'color' in kwargs:
        kwargs.pop('color')
        print("Warning: ignoring parameter 'color'. Use 'closses' and 'csurviving' instead.")
    if 'label' in kwargs:
        print("Warning: ignoring parameter 'label'.")
        kwargs.pop('label')

    if at_turn is None:
        at_turn=DA.max_turns
    if at_turn > DA.max_turns:
        raise ValueError(f'at_turn cannot be higher than the max number of turns for the simulation, here max_turn={DA.max_turns}')
    if DA.meta.nseeds==0:
        data = DA.survival_data.copy()
    else:
        data = DA.survival_data[DA.survival_data.seed==seed].copy()

    if type_plot=="polar":
        if "angle" not in data.columns or "amplitude" not in data.columns:
            data['angle']    = np.arctan2(data['y'],data['x'])*180/np.pi
            data['amplitude']= np.sqrt(data['x']**2+data['y']**2)

        if csurviving is not None and csurviving!='':
            surv=data.loc[data['nturns']>=at_turn,:]
            ax.scatter(surv['angle'], surv['amplitude'], color=csurviving, \
                       label="Surv.", **kwargs)

        if closses is not None and closses!='':
            loss=data.loc[data['nturns']<at_turn,:]
            if size_scaling in ["linear","log"]:
                if size_scaling=="linear":
                    size=(loss['nturns'].to_numpy()/at_turn) * plt.rcParams['lines.markersize']
                elif size_scaling=="log":
                    loss=loss.loc[loss['nturns']>0,:]
                    size=(np.log10(loss['nturns'].to_numpy())/np.log10(at_turn)) *\
                            plt.rcParams['lines.markersize']
                ax.scatter(loss['angle'], loss['amplitude'], size**2, color=closses, \
                           label="Loss.", **kwargs)
            else:
                ax.scatter(loss['angle'], loss['amplitude'], color=closses, \
                           label="Loss.", **kwargs)
            ax.set_xlabel(r'Angle [$^{\circ}$]')
            ax.set_ylabel(r'Amplitude [$\sigma$]')

    elif type_plot=="cartesian":
        if "x" not in data.columns or "y" not in data.columns:
            data['x']= data['amplitude']*np.cos(data['angle']*np.pi/180)
            data['y']= data['amplitude']*np.sin(data['angle']*np.pi/180)

        if csurviving is not None and csurviving!='':
            surv=data.loc[data['nturns']>=at_turn,:]
            ax.scatter(surv['x'], surv['y'], color=csurviving, label="Surv.", \
                       **kwargs)

        if closses is not None and closses!='':
            loss=data.loc[data['nturns']<at_turn,:]
            if size_scaling in ["linear","log"]:
                if size_scaling=="linear":
                    size=(loss['nturns'].to_numpy()/at_turn) * plt.rcParams['lines.markersize']
                elif size_scaling=="log":
                    loss=loss.loc[loss['nturns']>0,:]
                    size=(np.log10(loss['nturns'].to_numpy())/np.log10(at_turn)) *\
                            plt.rcParams['lines.markersize']
                ax.scatter(loss['x'], loss['y'], size**2, color=closses, label="Loss.", **kwargs)
            else:
                ax.scatter(loss['x'], loss['y'], color=closses, label="Loss.", **kwargs)
            ax.set_xlabel(r'x [$\sigma$]')
            ax.set_ylabel(r'y [$\sigma$]')

    else:
        raise ValueError('type_plot can only be either "polar" or "cartesian".')


def plot_da_border(DA, ax=None, at_turn=None, seed=None, type_plot="polar", clower="blue", cupper="red", **kwargs):
    """Plot the . border.

Parameters
----------
ax:        Plot axis.
at_turn:   All particles surviving at least this number of turns are considered as surviving.
seed:      In case of multiseed simulation, the seed number must be specified (Default=None).
type_plot: x-y for cartesian, ang-amp for polar (Default="polar").
clower:    Color of the lower DA estimation (Default="blue"). Use "" to disable.
cupper:    Color of the upper DA estimation (Default="red"). Use "" to disable.
"""

#         if DA.meta.pairs_shift != 0:
#             raise NotImplementedError("The DA computing methods have not been implemented for pairs yet!")

    if ax is None:
        ax = plt.subplots(1,1,figsize=(10,10))[1]

    if DA.meta.nseeds>0 and seed==None:
        raise ValueError('For multiseed simulation, please specify a seed number.')
    if seed=='stat':
        raise ValueError('"stat" border is not computed yet.')

    # Clean kwargs and initiallize parameters
    if 'c' in kwargs:
        kwargs.pop('c')
        print("Warning: ignoring parameter 'c'. Use 'closses' and 'csurviving' instead.")
    if 'color' in kwargs:
        kwargs.pop('color')
        print("Warning: ignoring parameter 'color'. Use 'closses' and 'csurviving' instead.")
    label = kwargs.pop('label', '')

    if at_turn is None:
        at_turn=DA.max_turns
    if at_turn > DA.max_turns:
        raise ValueError(f'at_turn cannot be higher than the max number of turns for the simulation, here max_turn={DA.max_turns}')
    if DA._border is None:
        DA.calculate_da(at_turn=at_turn,angular_precision=1,smoothing=True)

    data = DA.survival_data.copy()
    if "angle" not in data.columns:
        data['angle'] = np.arctan2(data['y'],data['x'])*180/np.pi
    else:
        data['angle'] = np.array(data.angle)
    angle=np.unique(data.angle)
    ang_range=(min(angle),max(angle))

    if DA.meta.nseeds==0:
        border=DA._border
    else:
        border=DA._border[DA._border.seed==seed]

    at_turn=max(border.t[border.t<=at_turn])

    mask_lower=(border.t==at_turn) & (~border['id lower'].isna())
    mask_upper=(border.t==at_turn) & (~border['id upper'].isna())
    fit_min=fit_DA(data.angle[border.loc[mask_lower,'id lower']],
                   data.amplitude[border.loc[mask_lower,'id lower']], ang_range)
    fit_max=fit_DA(data.angle[border.loc[mask_upper,'id upper']],
                   data.amplitude[border.loc[mask_upper,'id upper']], ang_range)

    angle  = np.sort(angle)
    amplitude_min=fit_min(angle)
    amplitude_max=fit_max(angle)
    extra_lab_lower = '' ; extra_lab_upper = '' ;
    if clower and cupper:
        extra_lab_lower = ' (min)' ; extra_lab_upper = ' (max)' ;
    
    if type_plot=="polar":
        if clower is not None and clower!='':
            ax.plot(angle,amplitude_min,color=clower,label=label+extra_lab_lower,**kwargs)

        if cupper is not None and cupper!='':
            ax.plot(angle,amplitude_max,color=cupper,label=label+extra_lab_upper,**kwargs)

        ax.set_xlabel(r'Angle [$^{\circ}$]')
        ax.set_ylabel(r'Amplitude [$\sigma$]')

    elif type_plot=="cartesian":
        if clower is not None and clower!='':
            x= amplitude_min*np.cos(angle*np.pi/180)
            y= amplitude_min*np.sin(angle*np.pi/180)
            ax.plot(x,y,color=clower,label=label+extra_lab_lower,**kwargs)

        if cupper is not None and cupper!='':
            x= amplitude_max*np.cos(angle*np.pi/180)
            y= amplitude_max*np.sin(angle*np.pi/180)
            ax.plot(x,y,color=cupper,label=label+extra_lab_upper,**kwargs)

        ax.set_xlabel(r'x [$\sigma$]')
        ax.set_ylabel(r'y [$\sigma$]')

    else:
        raise ValueError('type_plot can only be either "polar" or "cartesian".')


def plot_davsturns_border(DA, ax=None, from_turn=1e3, to_turn=None, y="DA", seed=None, clower="blue", cupper="red", 
                          show_Nm1=True, **kwargs): #show_seed=True, 
    """Plot the DA as a function of turns.

Parameters
----------
ax:        Plot axis.
from_turn: Lower turn range (Default: from_turn=1e3).
at_turn:   Upper turn range (Default: at_turn=max_turns).
seed:      In case of multiseed simulation, the seed number must be specified (Default=None).
clower:    Color of the lower da vs turns stat. Set to '' will not show the plot (Default: "blue").
cupper:    Color of the upper da vs turns stat. Set to '' will not show the plot (Default: "red").
show_seed: Plot seeds (Default: True).
show_Nm1:  Plot davsturns as a stepwise function (Default: True).
"""

    if ax is None:
        ax = plt.subplots(1,1,figsize=(10,10))[1]

    if DA.meta.nseeds>0 and seed==None:
        raise ValueError('For multiseed simulation, please specify the seed.')
    if DA.meta.nseeds==0 and seed=='stat':
        raise ValueError('"stat" is only available for multiseed simulation.')

    # Clean kwargs and initiallize parameters
    if 'c' in kwargs:
        kwargs.pop('c')
        print("Warning: ignoring parameter 'c'. Use 'closses' and 'csurviving' instead.")
    if 'color' in kwargs:
        kwargs.pop('color')
        print("Warning: ignoring parameter 'color'. Use 'closses' and 'csurviving' instead.")
    label = kwargs.pop('label', '')
    alpha = kwargs.pop('alpha', 1)

    if to_turn is None:
        to_turn=DA.max_turns
    if to_turn > DA.max_turns:
        raise ValueError(f'to_turn cannot be higher than the max number of turns for the simulation, here max_turn={DA.max_turns}')

    if DA.da is None:
        DA.calculate_davsturns(from_turn=from_turn,to_turn=to_turn)

    if DA.meta.nseeds==0:
        lower_da=DA._da.loc[:,["DA lower","DAmin lower","DAmax lower","t"]].set_index("t", drop=False)
        upper_da=DA._da.loc[:,["DA upper","DAmin upper","DAmax upper","t"]].set_index("t", drop=False)
    else:
        lower_da=DA._da.loc[DA._da.seed==seed,["DA lower","DAmin lower","DAmax lower","t"]].set_index("t", drop=False)
        upper_da=DA._da.loc[DA._da.seed==seed,["DA upper","DAmin upper","DAmax upper","t"]].set_index("t", drop=False)
        
    # Select the range of data
#     lturns_data=np.array([t for t in lower_da.turn if t>=from_turn and t<=to_turn])
    lturns_data=np.array([t for t in lower_da.t if t>=from_turn and t<=to_turn])
    lturns_data=lturns_data[np.argsort(lturns_data)]
    lturns_prev=[t-1 for t in lturns_data if t>from_turn and t<=to_turn]
    
    ycolumn=y.split(' ')
    if len(ycolumn)==2:
        show_estimate=ycolumn[1]
    else:
        show_estimate='both'
    ycolumn=ycolumn[0]

    if cupper is not None and cupper!='' and (show_estimate=="both" or show_estimate=="upper"):
        # Load Data
        davsturns_avg=upper_da.loc[lturns_data,ycolumn+' upper'] ;

        # Add step at turns-1
        if show_Nm1:
            for prev,turn in zip(lturns_prev, lturns_data[0:-1]):
                davsturns_avg[prev]=davsturns_avg[turn]

        lturns=np.array(davsturns_avg.index.tolist())
        lturns=lturns[np.argsort(lturns)]
        y_avg=np.array(davsturns_avg[lturns], dtype=float)

        # Plot the results
        llabel=label
        if label!='' and show_estimate=='both':
            llabel+=r" lower"
        ax.plot(lturns,y_avg,label=llabel,color=cupper,alpha=alpha,**kwargs);

#         if seed=='stat' and show_seed:
#             for s in range(1,DA.meta.nseeds+1):
#                 # Select the range of data
#                 slturns_data=np.array([t for t in DA._upper_davsturns[s].turn if t>=from_turn and t<=to_turn])
#                 slturns_data=slturns_data[np.argsort(slturns_data)]
#                 slturns_prev=[t-1 for t in slturns_data if t>from_turn and t<=to_turn]

#                 # Load Data
#                 davsturns_avg=DA._upper_davsturns[s].loc[slturns_data,'avg'] ;

#                 # Add step at turns-1
#                 if show_Nm1:
#                     for prev,turn in zip(slturns_prev, slturns_data[0:-1]):
#                         davsturns_avg[prev]=davsturns_avg[turn]

#                 lturns=np.array(davsturns_avg.index.tolist())
#                 lturns=lturns[np.argsort(lturns)]
#                 y_avg=np.array(davsturns_avg[lturns], dtype=float)

#                 # Plot the results
#                 ax.plot(lturns,y_avg,ls="-.",lw=1,label='',color=cupper,alpha=alpha*0.3,**kwargs);


    if clower is not None and clower!='' and (show_estimate=="both" or show_estimate=="lower"):
        # Load Data
        davsturns_avg=lower_da.loc[lturns_data,ycolumn+' lower'] ;

        # Add step at turns-1
        if show_Nm1:
            for prev,turn in zip(lturns_prev, lturns_data[0:-1]):
                davsturns_avg[prev]=davsturns_avg[turn]

        lturns=np.array(davsturns_avg.index.tolist())
        lturns=lturns[np.argsort(lturns)]
        y_avg=np.array(davsturns_avg[lturns], dtype=float)

        # Plot the results
        llabel=label
        if label!='' and show_estimate=='both':
            llabel+=r" upper"
        ax.plot(lturns,y_avg,label=llabel,color=clower,alpha=alpha,**kwargs);

#         if seed=='stat' and show_seed:
#             for s in range(1,DA.meta.nseeds+1):
#                 # Select the range of data
#                 slturns_data=np.array([t for t in DA._lower_davsturns[s].turn if t>=from_turn and t<=to_turn])
#                 slturns_data=slturns_data[np.argsort(slturns_data)]
#                 slturns_prev=[t-1 for t in slturns_data if t>from_turn and t<=to_turn]

#                 # Load Data
#                 davsturns_avg=DA._lower_davsturns[s].loc[slturns_data,'avg'] ;

#                 # Add step at turns-1
#                 if show_Nm1:
#                     for prev,turn in zip(slturns_prev, slturns_data[0:-1]):
#                         davsturns_avg[prev]=davsturns_avg[turn]

#                 lturns=np.array(davsturns_avg.index.tolist())
#                 lturns=lturns[np.argsort(lturns)]
#                 y_avg=np.array(davsturns_avg[lturns], dtype=float)

#                 # Plot the results
#                 ax.plot(lturns,y_avg,ls="-.",lw=1,label='',color=clower,alpha=alpha*0.3,**kwargs);

    ax.set_xlabel(r'Turns')
    ax.set_ylabel(r'Amplitude [$\sigma$]')
    
    


def plot_davsturns_extremum(DA, ax=None, from_turn=1e3, to_turn=None, seed=None, clower="blue", cupper="red", 
                          show_seed=True, show_Nm1=True, **kwargs):
    """Plot the DA as a function of turns.

Parameters
----------
ax:        Plot axis.
from_turn: Lower turn range (Default: from_turn=1e3).
at_turn:   Upper turn range (Default: at_turn=max_turns).
seed:      In case of multiseed simulation, the seed number must be specified (Default=None).
clower:    Color of the lower da vs turns stat. Set to '' will not show the plot (Default: "blue").
cupper:    Color of the upper da vs turns stat. Set to '' will not show the plot (Default: "red").
show_seed: Plot seeds (Default: True).
show_Nm1:  Plot davsturns as a stepwise function (Default: True).
"""

    if ax is None:
        ax = plt.subplots(1,1,figsize=(10,10))[1]

    if DA.meta.nseeds>0 and seed==None:
        raise ValueError('For multiseed simulation, please specify the seed.')
    if DA.meta.nseeds==0 and seed=='stat':
        raise ValueError('"stat" is only available for multiseed simulation.')

    # Clean kwargs and initiallize parameters
    if 'c' in kwargs:
        kwargs.pop('c')
        print("Warning: ignoring parameter 'c'. Use 'closses' and 'csurviving' instead.")
    if 'color' in kwargs:
        kwargs.pop('color')
        print("Warning: ignoring parameter 'color'. Use 'closses' and 'csurviving' instead.")
    if 'label' in kwargs:
        kwargs.pop('label')
        print("Warning: ignoring parameter 'label'.")
    alpha = kwargs.pop('alpha', 1)

    if to_turn is None:
        to_turn=DA.max_turns
    if to_turn > DA.max_turns:
        raise ValueError(f'to_turn cannot be higher than the max number of turns for the simulation, here max_turn={DA.max_turns}')

    if DA.da is None:
        DA.calculate_davsturns(from_turn=from_turn,to_turn=to_turn)

    if DA.meta.nseeds==0:
        lower_da=DA._da.loc[:,["DA lower","DAmin lower","DAmax lower","t"]].set_index("t", drop=False)
        upper_da=DA._da.loc[:,["DA upper","DAmin upper","DAmax upper","t"]].set_index("t", drop=False)
    else:
        lower_da=DA._da.loc[DA._da.seed==seed,["DA lower","DAmin lower","DAmax lower","t"]].set_index("t", drop=False)
        upper_da=DA._da.loc[DA._da.seed==seed,["DA upper","DAmin upper","DAmax upper","t"]].set_index("t", drop=False)
        
    # Select the range of data
    lturns_data=np.array([t for t in lower_da.t if t>=from_turn and t<=to_turn])
    lturns_data=lturns_data[np.argsort(lturns_data)]
    lturns_prev=[t-1 for t in lturns_data if t>from_turn and t<=to_turn]

    if cupper is not None and cupper!='':
        # Load Data
#         davsturns_avg=upper_da.loc[lturns_data,'DA upper'] ;
        davsturns_min=upper_da.loc[lturns_data,'DAmin upper'] ;
        davsturns_max=upper_da.loc[lturns_data,'DAmax upper'] ;

        # Add step at turns-1
        if show_Nm1:
            for prev,turn in zip(lturns_prev, lturns_data[0:-1]):
#                 davsturns_avg[prev]=davsturns_avg[turn]
                davsturns_min[prev]=davsturns_min[turn]
                davsturns_max[prev]=davsturns_max[turn]

        lturns=np.array(davsturns_min.index.tolist())
        lturns=lturns[np.argsort(lturns)]
#         y_avg=np.array(davsturns_avg[lturns], dtype=float)
        y_min=np.array(davsturns_min[lturns], dtype=float)
        y_max=np.array(davsturns_max[lturns], dtype=float)

        # Plot the results
        ax.plot(lturns,y_min,ls="-",   color=cupper,alpha=alpha,**kwargs);
        ax.plot(lturns,y_max,ls="-",   color=cupper,alpha=alpha,**kwargs);

        ax.fill_between(lturns,y_min, y_max,color=cupper,alpha=alpha*0.1,**kwargs)

        if seed=='stat' and show_seed:
            for s in range(1,DA.meta.nseeds+1):
                # Select the range of data
                supper_da=DA._da.loc[DA._da.seed==s,["DA upper","t"]].set_index("t", drop=False)
                slturns_data=np.array([t for t in supper_da.t if t>=from_turn and t<=to_turn])
                slturns_data=slturns_data[np.argsort(slturns_data)]
                slturns_prev=[t-1 for t in slturns_data if t>from_turn and t<=to_turn]

                # Load Data
                davsturns_avg=supper_da.loc[slturns_data,'DA upper'] ;

                # Add step at turns-1
                if show_Nm1:
                    for prev,turn in zip(slturns_prev, slturns_data[0:-1]):
                        davsturns_avg[prev]=davsturns_avg[turn]

                lturns=np.array(davsturns_avg.index.tolist())
                lturns=lturns[np.argsort(lturns)]
                y_avg=np.array(davsturns_avg[lturns], dtype=float)

                # Plot the results
                ax.plot(lturns,y_avg,ls="-.",lw=1,color=cupper,alpha=alpha*0.3,**kwargs);


    if clower is not None and clower!='':
        # Load Data
#         davsturns_avg=lower_da.loc[lturns_data,'DA lower'] ;
        davsturns_min=lower_da.loc[lturns_data,'DAmin lower'] ;
        davsturns_max=lower_da.loc[lturns_data,'DAmax lower'] ;

        # Add step at turns-1
        if show_Nm1:
            for prev,turn in zip(lturns_prev, lturns_data[0:-1]):
#                 davsturns_avg[prev]=davsturns_avg[turn]
                davsturns_min[prev]=davsturns_min[turn]
                davsturns_max[prev]=davsturns_max[turn]

        lturns=np.array(davsturns_min.index.tolist())
        lturns=lturns[np.argsort(lturns)]
#         y_avg=np.array(davsturns_avg[lturns], dtype=float)
        y_min=np.array(davsturns_min[lturns], dtype=float)
        y_max=np.array(davsturns_max[lturns], dtype=float)

        # Plot the results
        ax.plot(lturns,y_min,ls="-", color=clower,alpha=alpha,**kwargs);
        ax.plot(lturns,y_max,ls="-", color=clower,alpha=alpha,**kwargs);

        ax.fill_between(lturns,y_min, y_max,color=clower,alpha=alpha*0.1,**kwargs)

        if seed=='stat' and show_seed:
            for s in range(1,DA.meta.nseeds+1):
                # Select the range of data
                slower_da=DA._da.loc[DA._da.seed==s,["DA lower","t"]].set_index("t", drop=False)
                slturns_data=np.array([t for t in slower_da.t if t>=from_turn and t<=to_turn])
                slturns_data=slturns_data[np.argsort(slturns_data)]
                slturns_prev=[t-1 for t in slturns_data if t>from_turn and t<=to_turn]

                # Load Data
                davsturns_avg=slower_da.loc[slturns_data,'DA lower'] ;

                # Add step at turns-1
                if show_Nm1:
                    for prev,turn in zip(slturns_prev, slturns_data[0:-1]):
                        davsturns_avg[prev]=davsturns_avg[turn]

                lturns=np.array(davsturns_avg.index.tolist())
                lturns=lturns[np.argsort(lturns)]
                y_avg=np.array(davsturns_avg[lturns], dtype=float)

                # Plot the results
                ax.plot(lturns,y_avg,ls="-.",lw=1,color=clower,alpha=alpha*0.3,**kwargs);

    ax.set_xlabel(r'Turns')
    ax.set_ylabel(r'Amplitude [$\sigma$]')
