Pkg.add("RDatasets")
Pkg.add("Query")
Pkg.add("StatPlots")
Pkg.add("GR")

using RDatasets, Plots, Query, DataFrames

iris = dataset("datasets", "iris")

function SplitDFbySpecies(iris, species)
    df = @from i in iris begin
         @where i.Species == species
         @select {i.PetalLength, i.PetalWidth}
         @collect DataFrame
    end
    return df
end

function AttributeDescription(df, name)
    println(name, ":\n")
    colnames = names(df)
    ncols = size(colnames)[1]
    for i in 1:ncols
        println(colnames[i], ":")
        println(" mean: ", mean(df[i]))
        println(" var:  ", var(df[i]))
    end
    println()
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

function scale(df, iris)
    ncols = length(df.columns)
    means = dfmean(iris[[:PetalLength, :PetalWidth]])
    cov = dfcov(iris[[:PetalLength, :PetalWidth]])
    sigmas = [sqrt(cov[i, i]) for i in 1:ncols]
    
    scaler(x, mu, sigma) = (x .- mu) / sigma
    array = zeros(size(df))
    for i in 1:ncols
        array[:, i] = scaler(df[i], means[i], sigmas[i])
    end
    return array
end

setosa = SplitDFbySpecies(iris, "setosa")
versicolor = SplitDFbySpecies(iris, "versicolor")
virginica = SplitDFbySpecies(iris, "virginica")

gr()
scatter(setosa[1], setosa[2]; xlabel="Petal Length", ylabel="Petal Width",
        xlims=(1, 7), ylims=(-1, 4), m=(0.5, :s), label="setosa")
scatter!(versicolor[1], versicolor[2]; m=(0.5, :c), label="versicolor")
scatter!(virginica[1], virginica[2]; m=(0.5, :v), label="virginica")
savefig("figures/unscaled.png")

setosa_scaled = scale(setosa, iris)
versicolor_scaled = scale(versicolor, iris)
virginica_scaled = scale(virginica, iris)

scatter(setosa_scaled[:, 1], setosa_scaled[:, 2]; xlabel="Petal Length", ylabel="Petal Width",
        xlims=(-1.5, 1.5), ylims=(-1.5, 1.5), m=(0.5, :s), label="setosa")
scatter!(versicolor_scaled[:, 1], versicolor_scaled[:, 2]; m=(0.5, :c), label="versicolor")
scatter!(virginica_scaled[:, 1], virginica_scaled[:, 2]; m=(0.5, :v), label="virginica")
savefig("figures/scaled.png")