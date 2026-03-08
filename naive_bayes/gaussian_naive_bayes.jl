### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ de7255d5-6674-42cc-97ae-93dd86ada166
using Pkg

# ╔═╡ 7fa5a560-172f-11f1-b248-af1cbc3c97aa
Pkg.add("DelimitedFiles")

# ╔═╡ a524ecdd-5f2d-45dd-907b-fb534a3c7543
Pkg.add("Statistics")

# ╔═╡ b21cf563-353a-4d66-9a3f-76c780c6677c
using DelimitedFiles

# ╔═╡ 590f41c4-1c39-4aa2-863e-656e16ef9204
using(Statistics)

# ╔═╡ 082069f3-97ce-45df-aef6-a21bb55989b0
# đọc file
file_path = "C:\\julia\\naive_bayes\\data\\wdbc.txt"

# ╔═╡ b2e7c7a8-0e3c-4515-bea5-4c34096a096a
file_data = readdlm(file_path,',')

# ╔═╡ 5fd9149a-e3e5-4b97-9ac7-4fce2e3f0409
y = Int.(file_data[:,2]).+1

# ╔═╡ 184be301-ccf9-4f18-bc7f-960b60439def
X = file_data[:,3:end]

# ╔═╡ 5d801160-0834-4611-bfe1-c1db3faca9aa
function fit(X, y)
	K = length(unique(y))
	N,D = size(X)

	μ = zeros(D,K)
	σ = zeros(D,K)
	θ_k = zeros(K)

	for k = 1:K
		y_k = (y .==k)
		X_k = X[y_k,:]
		μ[:,k] = mean(X_k,dims=1)[:]
		σ[:,k] = std(X_k,dims=1)[:]
		θ_k[k] = sum(y_k)/length(y_k)
	end
	return μ,σ,θ_k
end

# ╔═╡ cd9f05b4-c76c-47ee-bd3a-1d7e7a5ae9d0
μ,σ,θ_k = fit(X,y)

# ╔═╡ 064d7b3a-eeb5-4ffd-8bb3-7d34c7ea90a7
function classify(x_new, μ, σ, θ_k)
    D, K = size(μ)
    log_prob = zeros(K)

    for k = 1:K
        log_prob[k] = sum(
            - (x_new .- μ[:,k]).^2 ./ (2 .* σ[:,k].^2)
            .- log.(σ[:,k])
            .- 0.5*log(2*pi)
        ) + log(θ_k[k])
    end

    return argmax(log_prob)
end

# ╔═╡ 0e7927c0-5182-4589-8ec0-97e97e3dd675
x_new = X[1,:]

# ╔═╡ ece870ef-2f5d-4077-8045-bdbae4bd823b
classify(x_new, μ, σ, θ_k)

# ╔═╡ ccc91a19-24eb-4869-b77b-e1935a141d2f
function accuracy(ground_truth, pred)
	return sum(ground_truth .== pred)/(length(pred))
end

# ╔═╡ 9f355b79-2139-4d61-a195-0345bcab365e
function evaluate(X_test,y_test, μ, σ, θ_k)
	N_test= length(y_test)
	pred = zeros(N_test)
	for i = 1:N_test
		pred[i] = classify(X_test[i,:],μ, σ, θ_k)
	end
	return accuracy(y_test,pred)
end

# ╔═╡ a5aaa25a-9a8a-4069-bebf-80c2c62963ff
evaluate(X,y,μ, σ, θ_k)

# ╔═╡ Cell order:
# ╠═de7255d5-6674-42cc-97ae-93dd86ada166
# ╠═7fa5a560-172f-11f1-b248-af1cbc3c97aa
# ╠═b21cf563-353a-4d66-9a3f-76c780c6677c
# ╠═082069f3-97ce-45df-aef6-a21bb55989b0
# ╠═b2e7c7a8-0e3c-4515-bea5-4c34096a096a
# ╠═5fd9149a-e3e5-4b97-9ac7-4fce2e3f0409
# ╠═184be301-ccf9-4f18-bc7f-960b60439def
# ╠═a524ecdd-5f2d-45dd-907b-fb534a3c7543
# ╠═590f41c4-1c39-4aa2-863e-656e16ef9204
# ╠═5d801160-0834-4611-bfe1-c1db3faca9aa
# ╠═cd9f05b4-c76c-47ee-bd3a-1d7e7a5ae9d0
# ╠═064d7b3a-eeb5-4ffd-8bb3-7d34c7ea90a7
# ╠═0e7927c0-5182-4589-8ec0-97e97e3dd675
# ╠═ece870ef-2f5d-4077-8045-bdbae4bd823b
# ╠═ccc91a19-24eb-4869-b77b-e1935a141d2f
# ╠═9f355b79-2139-4d61-a195-0345bcab365e
# ╠═a5aaa25a-9a8a-4069-bebf-80c2c62963ff
