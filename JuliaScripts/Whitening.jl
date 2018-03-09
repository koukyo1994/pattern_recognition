using Plots, RDatasets, DataFrames, StatPlots, Query

gr()

iris = dataset("datasets", "iris")

function dfmean(df::DataFrame)
    return [mean(i) for i in df.columns]
end

function dfcov(df)
    cols = df.columns
    ncols = size(cols)[1]
    ndatas = size(cols[1])[1]
    array = zeros(ncols, ncols)
    mean = dfmean(df)
    for i in 1:ncols
        for j in 1:ncols
            array[i, j] = 1/ndatas * (cols[i] .- mean[i])' * (cols[j] .- mean[j])
        end
    end
    return array
end

function whitening(df)
    groups = Array(df[:Species])
    petals = df[[:PetalLength, :PetalWidth]]
    colnames = names(petals)

    values = Array(petals)
    mu = reshape(dfmean(petals), (1, 2))

    coviris = dfcov(petals)
    S = -swap(eigvecs(coviris))

    Lambda = S' * coviris * S
    L = zeros(size(Lambda))
    for i in 1:size(L)[1]
        L[i, i] = sqrt(Lambda[i, i])
    end
    u = (values .- mu) * S * inv(L)

    tf_df = DataFrame()
    tf_df[colnames[1]] = u[:, 1]
    tf_df[colnames[2]] = u[:, 2]
    tf_df[:Species] = groups
    return tf_df
end

function swap(array)
    temp = zeros(2, 2)
    temp[:, 1] = array[:, 2]
    temp[:, 2] = array[:, 1]

    return temp
end

@df iris scatter(:PetalLength, :PetalWidth, group=:Species,
                 m=(0.5, [:+ :h :star7], 12), bg=RGB(.2,.2,.2))

scatter!(xlabel="Petal Length", ylabel="Petal Width", title="No Whitening",
         xlims=(1, 7), ylims=(-1, 4))
savefig("../figures/no_whitening.png")

whitened = whitening(iris)

@df whitened scatter(:PetalLength, :PetalWidth, group=:Species,
                         m=(0.5, [:+ :h :star7], 12), bg=RGB(.2,.2,.2))
scatter!(xlabel="Petal Length", ylabel="Petal Width", title="Whitened",
         xlims=(-3, 3), ylims=(-3, 3))
savefig("../figures/whitened.png")
