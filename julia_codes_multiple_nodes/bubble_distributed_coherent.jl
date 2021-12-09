# init libraries
using JLD2
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

# MPI not allowed
SL2Cfoam.is_MPI() && error("MPI version not allowed")

# argument parsing

CUTOFF = parse(Int, ARGS[3])
SHELLS = parse(Int, ARGS[4])

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
    
end

@eval @everywhere FOLDER = $(ARGS[2])
@eval @everywhere IMMIRZI = parse(Float64, $(ARGS[5]))

# init SL2Cfoam library
@everywhere begin

    conf = SL2Cfoam.Config(VerbosityOff, ACCURACY, 200, 0);
    SL2Cfoam.cinit(FOLDER, IMMIRZI, conf);

    # set C library automatic parallelization
    SL2Cfoam.set_OMP(OMP);

end

# logging function (flushing needed)
function log(x...)

    println("[ ", now(), " ] - ", join(x, " ")...)
    flush(stdout)
    
end

# bubble values
@everywhere begin

const onehalf = half(1);

# common boundary spin
const JB = Spin(3)

# set boundary
j12 = j13 = j14 = j15 = JB;

# build coherent state
angles = [     
                 0.0                  0.0
  1.9106332362490186                  0.0
  1.9106332362490186   2.0943951023931953
  1.9106332362490186  -2.0943951023931953
]
            
cstate = coherentstate_compute([ JB, JB, JB, JB ], angles)
            
end

# function that gives the bubble amplitude at fixed spins
@everywhere function bubble_amplitude_coherent(v::Vertex)

    dfj = prod([ dim(v.js[i]) for i in 5:10 ])
    
    vr = Vector{Float64}(undef, dim(JB))
    
    #TODO:  for loop needed?
    for i in 0:twice(JB)
    
    	vr[i+1] = dot(v.a[:,:,:,:,i+1], v.a[:,:,:,:,i+1])
    
    end
  
    dfj * only(contract(vr, cstate.a))

end

# runs over all internal spins
function bubble(cutoff, shells; step = onehalf)
    
    ampls = ComplexF64[]
    ampl = zero(ComplexF64)
    
    # loop over partial cutoffs
    for pcutoff = 0:step:cutoff
        
        # generate a list of all spins to compute
        spins_all = NTuple{10, HalfInt}[]
        for j23::HalfInt = 0:onehalf:pcutoff, j24::HalfInt = 0:onehalf:pcutoff, j25::HalfInt = 0:onehalf:pcutoff,
            j34::HalfInt = 0:onehalf:pcutoff, j35::HalfInt = 0:onehalf:pcutoff, j45::HalfInt = 0:onehalf:pcutoff
            
            # skip if computed in lower partial cutoff
            j23 <= (pcutoff-step) && j24 <= (pcutoff-step) &&
            j25 <= (pcutoff-step) && j34 <= (pcutoff-step) &&
            j35 <= (pcutoff-step) && j45 <= (pcutoff-step) && continue

            # skip if any intertwiner range empty
            r2, _ = intertwiner_range(j12, j25, j24, j23)
            r3, _ = intertwiner_range(j23, j13, j34, j35)
            r4, _ = intertwiner_range(j34, j24, j14, j45)
            r5, _ = intertwiner_range(j45, j35, j25, j15)

            isempty(r2) && continue
            isempty(r3) && continue
            isempty(r4) && continue
            isempty(r5) && continue

            # must be computed
            push!(spins_all, (j12, j13, j14, j15, j23, j24, j25, j34, j35, j45))

        end

        if isempty(spins_all)
            push!(ampls, 0.0)
            continue
        end

        # shuffle spins to equally distribute load
        shuffle!(spins_all)

        lt = @elapsed ampl += vertex_distribute(bubble_amplitude_coherent, spins_all, shells; store = (true, false))

        log("amplitude at partial cutoff $pcutoff: $ampl --- computed in $lt seconds")
        
        push!(ampls, ampl)
        
        # store partials 
        @save "$(FOLDER)/data/bubble-coherent__tj-$(twice(JB))__cutoff-$(CUTOFF)__shells-$(SHELLS)__imm-$(IMMIRZI).jld2" ampls
        
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


