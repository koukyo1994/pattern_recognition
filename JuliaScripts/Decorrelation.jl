using Plots, RDatasets, Query, DataFrames, StatPlots

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

function lintf(x, S)
    return S' * x
end

coviris = dfcov(iris[[:PetalLength, :PetalWidth]])
eigveciris = eigvecs(coviris)
y = lintf(iris[[:PetalLength, :PetalWidth]], S)

df = DataFrame(y)
@df iris scatter(:PetalLength, :PetalWidth, group=:Speicies)
