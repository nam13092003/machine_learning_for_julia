### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 701498d0-3de1-11f1-b73f-2d4463f6b181
using DelimitedFiles

# ╔═╡ 080ffc9c-c459-4d5b-b8fb-142bb8d90d97
using Optim

# ╔═╡ 89ea4233-eb95-4311-be73-8f5161b0f115
using Random

# ╔═╡ efe28de3-bd61-4573-a812-8b23fede47d7
data = readdlm("C:\\hus\\machine_learning\\julia\\softmax_regression\\data\\wine.data",',')

# ╔═╡ b8d40400-462b-4e27-9f2c-d78443d81bce
X = data[:,2:end]

# ╔═╡ 8da5650a-20e6-4a7f-8d07-15a4c9352413
y = Int.(data[:,1])

# ╔═╡ 67ed0fe1-614e-4006-8a1c-0a9065873234
θ_kd = rand(length(unique(y)),size(X,2))

# ╔═╡ e1e96a64-308e-48b9-ac91-9664e6657b13
y_onehot = zeros(length(unique(y)),size(X,1))

# ╔═╡ 3abf4b5e-c8a3-419f-874c-7514328dc6e3
for i in 1:length(y)
	y_onehot[y[i],i] =1
end

# ╔═╡ aa88a6f6-8340-4032-9fb0-854bc8583b07
y_onehot

# ╔═╡ 1254c670-80b8-4e5a-81f6-c005dc840afa
Z = θ_kd*X'

# ╔═╡ b0a36946-c1ca-4be4-b544-0cad3539d850
m = maximum(Z,dims=1)

# ╔═╡ 0d3b2949-e4bb-4beb-80aa-7f10f76c635f
θ_kd*X'.-m

# ╔═╡ e96691f3-bb77-4818-a07d-f6b0f6a4db62
sum(m .+ log.(sum(exp.(θ_kd*X'.-m),dims=1)))

# ╔═╡ 5cb363f8-8ec4-43f6-a5ab-abbee50a4222
sum(Z.*y_onehot)

# ╔═╡ 8b4b089c-f9f3-4689-bac0-4d2c111b5490
function J(X, y, θ_kd)
	Z = θ_kd*X'
	m = maximum(Z,dims=1)
	logsumexp = sum(m .+ log.(sum(exp.(θ_kd*X'.-m),dims=1) ))
	scores = sum(Z.*y)
	return -(scores -logsumexp)/length(y)
end

# ╔═╡ c1d4a615-2b62-480f-9640-7d1fc2324f6d
θ_init = ones(length(unique(y)),size(X,2))

# ╔═╡ 5b475c57-08c8-4d4a-ba5f-91a738290ac1
result = Optim.optimize(θ -> J(X, y_onehot,θ),θ_init, LBFGS())

# ╔═╡ aa9d86ff-80fb-40c5-9b52-d2991ac816f3
θ_best = Optim.minimizer(result)

# ╔═╡ 81601a04-fe77-4195-94b7-3e74baa4c393
function predict(X, θ)
	Z = θ*X' # kxn
	y_classify = zeros(size(X,1))
	for i in 1:length(y_classify)
		y_classify[i] = argmax(Z[:,i])
	end
	return y_classify
end

# ╔═╡ 6f54321f-3a84-4a68-b838-024e365b26f3
y_classify = predict(X, θ_best)

# ╔═╡ 1c4f6e7f-a80d-45b9-b842-98be9fb38cf9
sum(y .== y_classify)/length(y)

# ╔═╡ e02dbb82-79cb-4975-b783-0067cc540303
md"""
	sử dụng k-fold với k =5
"""

# ╔═╡ d66d170c-9f6a-47de-a4f8-7b9e39288916
function create_kfold(X, y,k)
	idx = randperm(size(X,1))
	folds = []
	size_fold = div(length(y),k)
	for i in 1:k
		start_idx = (i-1)*size_fold+1
		end_idx = i ==k ? length(y) : i* size_fold
		test_idx = idx[start_idx:end_idx]
		train_idx = setdiff(idx, test_idx)
		push!(folds,(train_idx,test_idx))
	
	end
	return folds
end

# ╔═╡ eb9bc040-f785-444b-8091-4bff3a66444b
function kfold_trainer(X, y,k)
	θ = zeros(length(unique(y)), size(X,2)) # khởi tạo theta của từng fold
	θ_init =  zeros(length(unique(y)), size(X,2)) # khởi tạo theta cho tối ưu
	for (train_idx, test_idx) in create_kfold(X, y, k)
		result = Optim.optimize(θ -> J(X[train_idx,:],y_onehot[:,train_idx],θ),θ_init,LBFGS())
		θ = θ .+ Optim.minimizer(result) # tính tổng tích lũy theo từng fold

	end
	return θ/k # tính trung bình theta
end

# ╔═╡ d852a308-1095-4ebf-b456-6bce000cc51f
θ_kfold = kfold_trainer(X, y,5)

# ╔═╡ da962c75-73b6-4ddf-b594-c657a18bbfd4
y_classify_kfold = predict(X, θ_kfold)

# ╔═╡ 6ba732d7-fe36-4ca1-a064-6b33cc0354e2
sum(y .== y_classify_kfold)/length(y)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Optim = "~2.0.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "905f5916a635dbd70a20beb2d7978f52538bfda6"

[[deps.ADTypes]]
git-tree-sha1 = "f7304359109c768cf32dc5fa2d371565bb63b68a"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.21.0"

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

    [deps.ADTypes.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "35ea197a51ce46fcd01c4a44befce0578a1aaeca"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.5.0"

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

    [deps.Adapt.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "78b3a7a536b4b0a747a0f296ea77091ca0a9f9a3"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.23.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceAMDGPUExt = "AMDGPU"
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = ["CUDSS", "CUDA"]
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceMetalExt = "Metal"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "7ae99144ea44715402c6c882bfef2adbeadbc4ce"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.16"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceGPUArraysCoreExt = "GPUArraysCore"
    DifferentiationInterfaceGTPSAExt = "GTPSA"
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = ["PolyesterForwardDiff", "ForwardDiff", "DiffResults"]
    DifferentiationInterfaceReverseDiffExt = ["ReverseDiff", "DiffResults"]
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceStaticArraysExt = "StaticArrays"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2f979084d1e13948a3352cf64a25df6bd3b4dca3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.16.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "9340ca07ca27093ff68418b7558ca37b05f8aeb1"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.29.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Printf"]
git-tree-sha1 = "738bdcacfef25b3a9e4a39c28613717a6b23751e"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.6.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.NLSolversBase]]
deps = ["ADTypes", "DifferentiationInterface", "FiniteDiff", "LinearAlgebra"]
git-tree-sha1 = "b3f76b463c7998473062992b246045e6961a074e"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "8.0.0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.Optim]]
deps = ["ADTypes", "EnumX", "FillArrays", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "PositiveFactorizations", "Printf", "SparseArrays", "Statistics"]
git-tree-sha1 = "7957b66b4e80f1031417197099f35273f7dd93dd"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "2.0.1"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"
"""

# ╔═╡ Cell order:
# ╠═701498d0-3de1-11f1-b73f-2d4463f6b181
# ╠═efe28de3-bd61-4573-a812-8b23fede47d7
# ╠═b8d40400-462b-4e27-9f2c-d78443d81bce
# ╠═8da5650a-20e6-4a7f-8d07-15a4c9352413
# ╠═67ed0fe1-614e-4006-8a1c-0a9065873234
# ╠═e1e96a64-308e-48b9-ac91-9664e6657b13
# ╠═3abf4b5e-c8a3-419f-874c-7514328dc6e3
# ╠═aa88a6f6-8340-4032-9fb0-854bc8583b07
# ╠═1254c670-80b8-4e5a-81f6-c005dc840afa
# ╠═b0a36946-c1ca-4be4-b544-0cad3539d850
# ╠═0d3b2949-e4bb-4beb-80aa-7f10f76c635f
# ╠═e96691f3-bb77-4818-a07d-f6b0f6a4db62
# ╠═5cb363f8-8ec4-43f6-a5ab-abbee50a4222
# ╠═8b4b089c-f9f3-4689-bac0-4d2c111b5490
# ╠═080ffc9c-c459-4d5b-b8fb-142bb8d90d97
# ╠═c1d4a615-2b62-480f-9640-7d1fc2324f6d
# ╠═5b475c57-08c8-4d4a-ba5f-91a738290ac1
# ╠═aa9d86ff-80fb-40c5-9b52-d2991ac816f3
# ╠═81601a04-fe77-4195-94b7-3e74baa4c393
# ╠═6f54321f-3a84-4a68-b838-024e365b26f3
# ╠═1c4f6e7f-a80d-45b9-b842-98be9fb38cf9
# ╠═e02dbb82-79cb-4975-b783-0067cc540303
# ╠═89ea4233-eb95-4311-be73-8f5161b0f115
# ╠═d66d170c-9f6a-47de-a4f8-7b9e39288916
# ╠═eb9bc040-f785-444b-8091-4bff3a66444b
# ╠═d852a308-1095-4ebf-b456-6bce000cc51f
# ╠═da962c75-73b6-4ddf-b594-c657a18bbfd4
# ╠═6ba732d7-fe36-4ca1-a064-6b33cc0354e2
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
