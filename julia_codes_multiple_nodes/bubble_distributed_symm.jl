# init libraries
using Random
using Distributed
using ClusterManagers
using Dates

length(ARGS) < 5 && error("args: ntasks folder cutoff shells Immirzi [thread_switch accuracy]")

# setup SLURM workers

ntasks = parse(Int, ARGS[1])
ClusterManagers.addprocs_slurm(ntasks; job_file_loc="/home/pfrisoni/log/juliacm")

# start distributed code

@everywhere using SL2Cfoam
@everywhere using HalfIntegers
@everywhere using LinearAlgebra
@everywhere using JLD2

# MPI not allowed
SL2Cfoam.is_MPI() && error("MPI version not allowed")

# argument parsing

CUTOFF = HalfInt(parse(Float64, ARGS[3]))
SHELLS = parse(Int, ARGS[4])

PCUTOFF0 = HalfInt(0)
AMPL0 = 0.0

@everywhere OMP = true
@everywhere ACCURACY = HighAccuracy

if length(ARGS) > 5

    if ARGS[6] == "thread_OFF"
        @everywhere OMP = false
    end
    
    if ARGS[7] == "accuracy_NORMAL"
        @everywhere ACCURACY = NormalAccuracy
    elseif ARGS[7] == "accuracy_VERYHIGH"
    	@everywhere ACCURACY = VeryHighAccuracy
    end
    
    if length(ARGS) > 7
    	PCUTOFF0 = HalfInt(parse(Float64, ARGS[8]))
    	AMPL0 = parse(Float64, ARGS[9])
    end
    
end

@eval @everywhere FOLDER = $(ARGS[2])
@eval @everywhere IMMIRZI = parse(Float64, $(ARGS[5]))

# init SL2Cfoam library
@everywhere begin

    conf = SL2Cfoam.Config(VerbosityOff, ACCURACY, 200, 0);
    SL2Cfoam.cinit(FOLDER, IMMIRZI, conf);

    # set C library automatic parallelization
    SL2Cfoam.set_OMP(OMP);

    # set boundary
    const onehalf = half(1);
    j12 = j13 = j14 = j15 = onehalf;
    i1 = zero(HalfInt);

end

# logging function (flushing needed)
function log(x...)

    println("[ ", now(), " ] - ", join(x, " ")...)
    flush(stdout)
    
end

# runs over all internal spins
function bubble(cutoff, shells; step = onehalf)
    
    ampls = Float64[]
    ampl = AMPL0
    
    # loop over partial cutoffs
    for pcutoff = PCUTOFF0:step:cutoff
        
        @everywhere pcutstr = string(Float64($pcutoff))
        @everywhere @load "/scratch/pfrisoni/bubble/spins_configurations/spins_all_symmetries__CUTOFF-$pcutstr.jld2" spins_all

        if isempty(spins_all)
            push!(ampls, 0.0)
            continue
        end

        @everywhere function bubble_amplitude_symm(v::Vertex)

            # find the symmetry factor in the list
            bjs = v.js[5:10]
            sind = findfirst(js -> js[1:6] == bjs, spins_all)
            sf = spins_all[sind][7]

            dfj = prod([ dim(v.js[i]) for i in 5:10 ])
        
            dfj * dot(v.a[:,:,:,:,1], v.a[:,:,:,:,1]) * sf

        end

        # shuffle spins to equally distribute load
        shuffle!(spins_all)

        # boring: I need to add back the fixed spins
        spins_all_prop = Vector{NTuple{10, Spin}}(undef, length(spins_all))
        for (i, js) in enumerate(spins_all)
            spins_all_prop[i] = (j12, j13, j14, j15, js[1:6]...)
        end

        lt = @elapsed ampl += vertex_distribute(bubble_amplitude_symm, spins_all_prop, shells; store = (true, false))

        log("amplitude at partial cutoff $pcutoff: $ampl --- computed in $lt seconds")
        
        push!(ampls, ampl)
        
        # store partials 
        if PCUTOFF0 == 0
            @save "$(FOLDER)/data/bubble__cutoff-$(float(CUTOFF))__shells-$(SHELLS)__imm-$(IMMIRZI).jld2" ampls
        else
            @save "$(FOLDER)/data/bubble__cutoff-$(float(PCUTOFF0)):$(float(CUTOFF))__shells-$(SHELLS)__imm-$(IMMIRZI).jld2" ampls
        end
        
    end # partial cutoffs loop
    
    ampls
    
end

log("master process is on node $(gethostname())")

@time ampls = bubble(CUTOFF, SHELLS);

@show ampls

# release workers

for i in workers()
	rmprocs(i)
end

println("Completed.")


