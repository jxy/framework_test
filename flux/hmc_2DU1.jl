using Random
using Flux.Tracker

struct Param
    beta
    lat
    nd
    volume
    tau
    nstep
    dt
    ntraj
    nrun
    nprint
    seed
    randinit
end

function Param(;
        beta = 6.0, lat = [64, 64],
        tau = 2.0, nstep = 50, ntraj = 256, nrun = 4, nprint = 256,
        seed = 11*13, randinit = false)
    nd = length(lat)
    volume = prod(lat)
    dt = tau / nstep
    Param(beta,lat,nd,volume,tau,nstep,dt,ntraj,nrun,nprint,seed,randinit)
end

summary(param::Param) = """latsize = $(param.lat)
    volume = $(param.volume)
    beta = $(param.beta)
    trajs = $(param.ntraj)
    tau = $(param.tau)
    steps = $(param.nstep)
    seed = $(param.seed)
    """
function uniquestr(param::Param)
    lat = join([string(x) for x in param.lat], ".")
    "out_l$(lat)_b$(param.beta)_n$(param.ntraj)_t$(param.tau)_s$(param.nstep).out"
end

action(param, f) = (-param.beta)*sum(cos.(plaqphase(f)))
force(param, f) = Tracker.gradient(x -> action(param, x), f)[1]

# Flux cannot take the derivatives of the circshift
#plaqphase(f) = f[1,:,:] - f[2,:,:] - circshift(f[1,:,:], [0,-1]) + circshift(f[2,:,:], [-1,0])
@views plaqphase(f) = f[1,:,:] - f[2,:,:] - hcat(f[1,:,2:end], f[1,:,1:1]) + vcat(f[2,2:end,:], f[2,1:1,:])

topocharge(f) = round(sum(regularize(plaqphase(f))) / 2pi)
function regularize(f)
    p2 = 2pi
    # f_ = f .- pi
    p2 .* ((f .- pi) ./ p2 .- fld.((f .- pi), p2) .- 0.5)
end

function leapfrog(param, x, p)
    dt = param.dt
    x_ = x + 0.5dt .* p
    p_ = p + (-dt) .* Tracker.data(force(param, x_))
    for i = 1:param.nstep
        x_ += dt .* p_
        p_ += (-dt) .* Tracker.data(force(param, x_))
    end
    x_ += 0.5dt .* p_
    (x_, p_)
end
function hmc(param, x)
    p = randn(eltype(x), size(x))
    act0 = action(param, x) + 0.5*sum(p .* p)
    x_, p = leapfrog(param, x, p)
    x_ = regularize(x_)
    act = action(param, x_) + 0.5*sum(p .* p)
    prob = rand()
    dH = act-act0
    exp_mdH = exp(-dH)
    acc = prob < exp_mdH
    newx = acc ? x_ : x
    (dH, exp_mdH, acc, newx)
end

put(s) = write(stdout, s)

function main()
    param = Param(
        beta = 6.0, # 4.0
        lat = [64, 64], # [8, 8]
        tau = 2, # 0.3
        nstep = 50, # 3
        ntraj = 256, # 2**16 # 2**10 # 2**15
        nprint = 256,
        seed = 1331)

    Random.seed!(param.seed)

    field = param.randinit ? randn(Float64, (param.nd, param.lat...)) : zeros(Float64, (param.nd, param.lat...))  # mu, x, t

    open(uniquestr(param), "w") do O
        params = summary(param)
        write(O, params)
        put(params)
        plaq_, topo_ = (action(param, field) / (-param.beta*param.volume), topocharge(field))
        status = "Initial configuration:  plaq: $(plaq_)  topo: $(topo_)\n"
        write(O, status)
        put(status)
        ts = Float64[]
        for n = 1:param.nrun
            t = -time()
            for i = 1:param.ntraj
                dH, exp_mdH, acc, field = hmc(param, field)
                plaq = action(param, field) / (-param.beta*param.volume)
                topo = topocharge(field)
                ifacc = acc ? "ACCEPT" : "REJECT"
                status = "Traj: $((n-1)*param.ntraj+i) $ifacc:  dH: $dH  exp(-dH): $exp_mdH  plaq: $plaq  topo: $topo\n"
                write(O, status)
                if i % div(param.ntraj, param.nprint) == 0
                    put(status)
                end
            end
            t += time()
            append!(ts, t)
        end
        print("Run times: $ts\n")
        print("Per trajectory: $([t/param.ntraj for t in ts])\n")
    end
end

main()
