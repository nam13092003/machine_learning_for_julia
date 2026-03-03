### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 322ce26a-1818-4865-a9dd-e12aa9997475
using DelimitedFiles

# ╔═╡ a073661f-c60d-4edf-a2b5-8fd339943a49
md"""
Khởi tạo dữ liệu huấn luyện
"""

# ╔═╡ da689717-9da4-4d92-839a-f94b228d05c7
X = [1 1 1 0;
	 1 1 1 1;
	0 1 1 0;
	 0 0 1 1;
	 0 0 0 0;
	 0 0 0 1;
	 1 0 0 0;
	0 1 1 1
	]

# ╔═╡ 1e5370e3-4c2e-4916-b037-6d0bffd37352
y = [0,0,1, 1, 1, 0, 1, 0] .+1

# ╔═╡ 56368fb5-9751-4291-86e4-94b9e0ad8210
md"""
Hàm huấn luyện mô hình
"""

# ╔═╡ 65e442f8-ef6e-456b-a505-380b1658d5b8
function fit(X, y)
    K = length(unique(y))
    N, D = size(X)
    θ_k = zeros(K)
    θ_jk = zeros(D, K)
    for k in 1:K
        b_k = (y .== k)
        θ_k[k] = sum(b_k) / N
        X_k = X[b_k, :]
        θ_jk[:, k] = sum(X_k, dims=1)[:] / sum(b_k)
    end
    return θ_k, θ_jk
end

# ╔═╡ f79df737-027f-4a2f-9d86-7a9fea0a2d79
θ_k, θ_jk = fit(X,y)

# ╔═╡ abc695ec-6c77-44e1-bd60-0e31a0eb8498
function classify(x_new, θ_k, θ_jk)
    K = length(θ_k)
    log_probs = zeros(Float64, K)

    for k in 1:K
        log_likelihood = sum(
            x_new .* log.(θ_jk[:, k]) +
            (1 .- x_new) .* log.(1 .- θ_jk[:, k])
        )

        log_probs[k] = log(θ_k[k]) + log_likelihood
    end

    return argmax(log_probs)
end

# ╔═╡ 3cf15132-1fe3-404a-93c6-706a8df859d4
x_new = [1,0,1,0]

# ╔═╡ 04e4964b-4eda-42d2-9bcb-d708bae1b793
classify(x_new, θ_k, θ_jk)

# ╔═╡ 0ba54690-d06f-4542-9441-f8aa9585ecd3
md"""
Hướng dẫn đọc dữ liệu 
"""

# ╔═╡ 4b0b7a75-e370-4ec6-944a-b68718aa67f2
feature_path_train = "C:/Users/dangn/Downloads/ex6DataPrepared/train-features.txt"

# ╔═╡ 2d9f9f54-9b31-4362-86a6-e089246dbe62
label_path_train = "C:/Users/dangn/Downloads/ex6DataPrepared/train-labels.txt"

# ╔═╡ ed0d21b9-f731-4702-ba88-9444efdab940
function read_data(feature_path, label_path)
	pre_data = Int.(readdlm(feature_path))
	N = length(unique(pre_data[:,1]))
	D = 2500
	train_data = zeros(Int,N,D)
	for i in 1:N
		for i in 1:D
			train_data[pre_data[i,1],pre_data[i,2]] = 1
		end
	end
	y_train = Int.(readdlm(label_path))[:] .+1
	return train_data, y_train
end

# ╔═╡ 8c81514a-9b7f-45bb-a7a1-1ad6ce552d81
X_train,y_train = read_data(feature_path_train,label_path_train)

# ╔═╡ 0b245840-a1ad-45b1-b217-7b2bbb7294a5
md"""
thử nghiệm với hàm fit chưa dùng laplace smoothing
"""

# ╔═╡ f049c7e8-2ea3-428c-9035-93102336798b
θ_k_spam ,θ_jk_spam = fit(X_train,y_train)

# ╔═╡ c0e1d7ff-8fb7-4367-9093-265016437d21
feature_path_test = "C:/Users/dangn/Downloads/ex6DataPrepared/test-features.txt"

# ╔═╡ edf95cd7-3b3f-480f-bfa5-ab9332eeece2
label_path_test = "C:/Users/dangn/Downloads/ex6DataPrepared/test-labels.txt"

# ╔═╡ c98ce82d-d963-483b-8230-993be5e8257c
X_test,y_test = read_data(feature_path_test,label_path_test)

# ╔═╡ 29c58b7d-782f-44f8-8221-d3b2b33ea657
function accuracy(grouth_truth, pred)
    correct = sum(grouth_truth .== pred)
    return correct / length(grouth_truth)
end

# ╔═╡ bc569bff-b821-4a7b-beba-0c8676090fb9
function evaluate(X_test,y_test,θ_k, θ_jk)
	N_test = length(y_test)
	pred = zeros(Int,N_test)
	for i in 1:N_test
		pred[i] = classify(X_test[i,:],θ_k, θ_jk)
	end
	return accuracy(y_test,pred)
end
	

# ╔═╡ b658d247-113c-4d46-a13d-278a975e78ca
evaluate(X_train,y_train,θ_k_spam ,θ_jk_spam)

# ╔═╡ ae1659f3-05b7-4491-818a-349e5e400faa
evaluate(X_test,y_test,θ_k_spam ,θ_jk_spam)

# ╔═╡ b9dd1a8b-3b15-42ee-a504-29f3c6cc627f
md"""
thử nghiệm với hàm fit dùng laplace smoothing
"""

# ╔═╡ d2599c75-c3cb-4591-9077-569dfc65719d
function fit_laplace_smoothing(X, y)
    K = length(unique(y))
    N, D = size(X)
    θ_k = zeros(K)
    θ_jk = zeros(D, K)
    for k in 1:K
        b_k = (y .== k)
        θ_k[k] = sum(b_k) / N
        X_k = X[b_k, :]
        θ_jk[:, k] = (sum(X_k, dims=1)[:] .+1) / (sum(b_k)+2)
    end
    return θ_k, θ_jk
end

# ╔═╡ b08dd0c8-f207-4b2d-b29b-7b7ecfadfdbd
θ_k_spam_ls ,θ_jk_spam_ls = fit_laplace_smoothing(X_train,y_train)

# ╔═╡ de0f3475-9dca-4210-9bf6-d6d886f61416
evaluate(X_train,y_train,θ_k_spam_ls ,θ_jk_spam_ls)

# ╔═╡ 9844d1ca-eea9-4349-8857-53ca065620af
evaluate(X_test,y_test,θ_k_spam_ls ,θ_jk_spam_ls)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "c2bd3517fa16afe10381a499d946223d1765af60"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"
"""

# ╔═╡ Cell order:
# ╠═a073661f-c60d-4edf-a2b5-8fd339943a49
# ╠═da689717-9da4-4d92-839a-f94b228d05c7
# ╠═1e5370e3-4c2e-4916-b037-6d0bffd37352
# ╠═56368fb5-9751-4291-86e4-94b9e0ad8210
# ╠═65e442f8-ef6e-456b-a505-380b1658d5b8
# ╠═f79df737-027f-4a2f-9d86-7a9fea0a2d79
# ╠═abc695ec-6c77-44e1-bd60-0e31a0eb8498
# ╠═3cf15132-1fe3-404a-93c6-706a8df859d4
# ╠═04e4964b-4eda-42d2-9bcb-d708bae1b793
# ╠═0ba54690-d06f-4542-9441-f8aa9585ecd3
# ╠═322ce26a-1818-4865-a9dd-e12aa9997475
# ╠═4b0b7a75-e370-4ec6-944a-b68718aa67f2
# ╠═2d9f9f54-9b31-4362-86a6-e089246dbe62
# ╠═ed0d21b9-f731-4702-ba88-9444efdab940
# ╠═8c81514a-9b7f-45bb-a7a1-1ad6ce552d81
# ╠═0b245840-a1ad-45b1-b217-7b2bbb7294a5
# ╠═f049c7e8-2ea3-428c-9035-93102336798b
# ╠═c0e1d7ff-8fb7-4367-9093-265016437d21
# ╠═edf95cd7-3b3f-480f-bfa5-ab9332eeece2
# ╠═c98ce82d-d963-483b-8230-993be5e8257c
# ╠═29c58b7d-782f-44f8-8221-d3b2b33ea657
# ╠═bc569bff-b821-4a7b-beba-0c8676090fb9
# ╠═b658d247-113c-4d46-a13d-278a975e78ca
# ╠═ae1659f3-05b7-4491-818a-349e5e400faa
# ╠═b9dd1a8b-3b15-42ee-a504-29f3c6cc627f
# ╠═d2599c75-c3cb-4591-9077-569dfc65719d
# ╠═b08dd0c8-f207-4b2d-b29b-7b7ecfadfdbd
# ╠═de0f3475-9dca-4210-9bf6-d6d886f61416
# ╠═9844d1ca-eea9-4349-8857-53ca065620af
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
