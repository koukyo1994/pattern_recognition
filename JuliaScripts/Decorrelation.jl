using Plots, RDatasets, Query, DataFrames, StatPlots

gr()

iris = dataset("datasets", "iris")

function SplitDFbySpecies(iris, species)
    df = @from i in iris begin
         @where i.Species == species
         @select {i.PetalLength, i.PetalWidth}
         @collect DataFrame
    end
    return df
end

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

function lintf(df, S)
    groups = Array(df[:Species])
    petals = df[[:PetalLength, :PetalWidth]]

    colnames = names(petals)
    values = Array(petals)
    values = values * S
    tf_df = DataFrame()
    tf_df[colnames[1]] = values[:, 1]
    tf_df[colnames[2]] = values[:, 2]
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

scatter!(xlabel="Petal Length", ylabel="Petal Width", title="No Decorrelation",
         xlims=(1, 7), ylims=(-1, 4))
savefig("../figures/no_decorrelation.png")

coviris = dfcov(iris[[:PetalLength, :PetalWidth]])
S = -swap(eigvecs(coviris))

decorrelated = lintf(iris, S)
covdecorrelated = dfcov(decorrelated[[:PetalLength, :PetalWidth]])

@df decorrelated scatter(:PetalLength, :PetalWidth, group=:Species,
                         m=(0.5, [:+ :h :star7], 12), bg=RGB(.2,.2,.2))
scatter!(xlabel="Petal Length", ylabel="Petal Width", title="Decorrelated",
         xlims=(1, 7), ylims=(-2, 3))
savefig("../figures/decorrelation.png")
