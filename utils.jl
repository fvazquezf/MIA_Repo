module Utils

    using Flux
    using Flux.Losses
    using Statistics
    using Random
    using Printf

    # ---------------------------------------------------------------------------
    # --- From: Unit 2 - Multilayer Perceptron & Unit 4.2 - Multiclass classification ---
    # ---------------------------------------------------------------------------

    """
        oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})

    Performs one-hot encoding for a given feature vector.

    - If there are 2 classes, it returns a boolean matrix with one column.
    - If there are more than 2 classes, it returns a boolean matrix with one column per class.
    """
    function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
        @assert(all([in(value, classes) for value in feature]));
        numClasses = length(classes);
        @assert(numClasses > 1)

        if (numClasses == 2)
            # Case with only two classes
            oneHot = reshape(feature .== classes[1], :, 1);
        else
            # Case with more than two classes
            oneHot = BitArray{2}(undef, length(feature), numClasses);
            for numClass = 1:numClasses
                oneHot[:, numClass] .= (feature .== classes[numClass]);
            end;
        end;
        return oneHot;
    end;

    # Overloaded methods for oneHotEncoding
    oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));
    oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1);


    # ---------------------------------------------------------------------------
    # --- From: Unit 2 - Multilayer Perceptron ---
    # ---------------------------------------------------------------------------

    """
        calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})

    Calculates the minimum and maximum values for each column in a dataset.
    Returns a tuple containing two 1-row matrices: (min_values, max_values).
    """
    function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
        return minimum(dataset, dims=1), maximum(dataset, dims=1)
    end;

    """
        calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})

    Calculates the mean and standard deviation for each column in a dataset.
    Returns a tuple containing two 1-row matrices: (mean_values, std_values).
    """
    function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
        return mean(dataset, dims=1), std(dataset, dims=1)
    end;

    """
        normalizeMinMax!(dataset, normalizationParameters)

    Performs min-max normalization on the dataset in-place.
    """
    function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
        minValues = normalizationParameters[1];
        maxValues = normalizationParameters[2];
        dataset .-= minValues;
        range = maxValues .- minValues;
        # Handle columns where min and max are the same to avoid division by zero
        range[range .== 0] .= 1;
        dataset ./= range;
        # Set columns with no variance to 0
        dataset[:, vec(minValues.==maxValues)] .= 0;
        return dataset;
    end;

    normalizeMinMax!(dataset::AbstractArray{<:Real,2}) = normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset));

    """
        normalizeMinMax(dataset, normalizationParameters)

    Performs min-max normalization on a copy of the dataset.
    """
    function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
        return normalizeMinMax!(copy(dataset), normalizationParameters);
    end;

    normalizeMinMax(dataset::AbstractArray{<:Real,2}) = normalizeMinMax!(copy(dataset));


    """
        normalizeZeroMean!(dataset, normalizationParameters)

    Performs zero-mean (standardization) normalization on the dataset in-place.
    """
    function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
        avgValues = normalizationParameters[1];
        stdValues = normalizationParameters[2];
        dataset .-= avgValues;
        # Handle columns with zero standard deviation
        stdValues[stdValues .== 0] .= 1;
        dataset ./= stdValues;
        # Set columns with no variance to 0
        dataset[:, vec(stdValues .== 0)] .= 0;
        return dataset;
    end;

    normalizeZeroMean!(dataset::AbstractArray{<:Real,2}) = normalizeZeroMean!(dataset, calculateZeroMeanNormalizationParameters(dataset));

    """
        normalizeZeroMean(dataset, normalizationParameters)

    Performs zero-mean (standardization) normalization on a copy of the dataset.
    """
    function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
        return normalizeZeroMean!(copy(dataset), normalizationParameters);
    end;

    normalizeZeroMean(dataset::AbstractArray{<:Real,2}) = normalizeZeroMean!(copy(dataset));


    """
        classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)

    Converts model outputs (probabilities or logits) to boolean class predictions.
    - For a single output column, applies a threshold.
    - For multiple columns, selects the class with the maximum value (argmax).
    """
    function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
        numOutputs = size(outputs, 2);
        @assert(numOutputs != 2) 

        if numOutputs == 1
            return outputs .>= threshold;
        else
            (_, indicesMaxEachInstance) = findmax(outputs, dims=2);
            classified_outputs = falses(size(outputs));
            classified_outputs[indicesMaxEachInstance] .= true;
            @assert(all(sum(classified_outputs, dims=2) .== 1));
            return classified_outputs;
        end;
    end;


    """
        accuracy(outputs, targets)

    Calculates the classification accuracy. Overloaded for different input types.
    """
    accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) = mean(outputs .== targets);

    function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
        @assert(size(outputs) == size(targets));
        if (size(targets, 2) == 1)
            return accuracy(view(outputs, :, 1), view(targets, :, 1));
        else
            # For one-hot, check if all elements in each row match
            return mean(all(targets .== outputs, dims=2));
        end;
    end;

    accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) = accuracy(outputs .>= threshold, targets);

    function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
        @assert(size(outputs) == size(targets));
        if (size(targets, 2) == 1)
            return accuracy(view(outputs, :, 1), view(targets, :, 1), threshold=threshold);
        else
            return accuracy(classifyOutputs(outputs), targets);
        end;
    end;


    # ---------------------------------------------------------------------------
    # --- From: Unit 3 - Overfitting ---
    # ---------------------------------------------------------------------------

    """
        holdOut(N::Int, P::Real)

    Splits N indices into two disjoint sets for training and testing.
    P is the percentage of patterns for the test set (e.g., 0.2 for 20%).
    """
    function holdOut(N::Int, P::Real)
        @assert 0.0 <= P <= 1.0 "P must be in [0,1]"
        @assert N > 0 "N must be > 0"
        idx = randperm(N)
        n_test = Int(round(P * N))
        return (idx[n_test+1:end], idx[1:n_test]) # (train_idx, test_idx)
    end

    """
        holdOut(N::Int, Pval::Real, Ptest::Real)

    Splits N indices into three disjoint sets: training, validation, and testing.
    Pval and Ptest are the percentages for validation and test sets, respectively.
    """
    function holdOut(N::Int, Pval::Real, Ptest::Real)
        @assert 0.0 <= Pval <= 1.0 "Pval must be in [0,1]"
        @assert 0.0 <= Ptest <= 1.0 "Ptest must be in [0,1]"
        @assert (Pval + Ptest) <= 1.0 "Pval + Ptest must be <= 1"

        (train_val_idx, test_idx) = holdOut(N, Ptest)
        
        # Adjust validation percentage for the remaining data
        Pval_adjusted = Pval / (1 - Ptest)
        (train_idx_rel, val_idx_rel) = holdOut(length(train_val_idx), Pval_adjusted)
        
        train_idx = train_val_idx[train_idx_rel]
        val_idx = train_val_idx[val_idx_rel]
        
        return (train_idx, val_idx, test_idx)
    end


    # ---------------------------------------------------------------------------
    # --- From: Unit 2 & 3 (Combined & Updated) ---
    # ---------------------------------------------------------------------------

    """
        buildClassANN(numInputs, topology, numOutputs; transferFunctions)

    Builds a Flux Chain (ANN) for a classification problem.
    - `numOutputs=1` creates a binary classifier with a sigmoid output.
    - `numOutputs>2` creates a multi-class classifier with a softmax output.
    """
    function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
        ann = Chain();
        numInputsLayer = numInputs;
        for (i, numHiddenLayer) in enumerate(topology)
            ann = Chain(ann..., Dense(numInputsLayer, numHiddenLayer, transferFunctions[i]));
            numInputsLayer = numHiddenLayer;
        end;

        if (numOutputs == 1)
            ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
        else
            ann = Chain(ann..., Dense(numInputsLayer, numOutputs)); # No activation, softmax is added by the loss function
            ann = Chain(ann..., softmax);
        end;
        return ann;
    end;

    """
        trainClassANN(topology, trainingDataset; validationDataset, testDataset, ...)

    Trains a classification ANN with early stopping.

    Returns the trained model, and the history of training, validation, and test losses.
    """
    function trainClassANN(
            topology::AbstractArray{<:Int,1},
            trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
            validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}} = (Array{Float32}(undef,0,0), falses(0,0)),
            testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}} = (Array{Float32}(undef,0,0), falses(0,0)),
            transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
            maxEpochs::Int=1000,
            minLoss::Real=0.0,
            learningRate::Real=0.01,
            maxEpochsVal::Int=20
        )

        (trainingInputs, trainingTargets) = trainingDataset
        (validationInputs, validationTargets) = validationDataset
        (testInputs, testTargets) = testDataset

        @assert size(trainingInputs,1) == size(trainingTargets,1)

        ann = buildClassANN(size(trainingInputs,2), topology, size(trainingTargets,2), transferFunctions=transferFunctions)
        
        loss(model,x,y) = (size(y, 2) == 1) ? binarycrossentropy(model(x), y) : crossentropy(model(x), y)

        
        opt_state = Flux.setup(Adam(learningRate), ann)

        trainingLosses = Float64[]
        validationLosses = Float64[]
        testLosses = Float64[]

        bestValidationLoss = Inf
        epochs_no_improve = 0
        best_ann = deepcopy(ann)

        # Epoch 0: Initial losses
        push!(trainingLosses, loss(ann, trainingInputs', trainingTargets'))
        !isempty(validationInputs) && push!(validationLosses, loss(ann, validationInputs', validationTargets'))
        !isempty(testInputs) && push!(testLosses, loss(ann, testInputs', testTargets'))


        for epoch in 1:maxEpochs
            Flux.train!(loss, ann, [(trainingInputs', trainingTargets')], opt_state)
            
            current_train_loss = loss(ann, trainingInputs', trainingTargets')
            push!(trainingLosses, current_train_loss)

            if !isempty(validationInputs)
                current_val_loss = loss(ann, validationInputs', validationTargets')
                push!(validationLosses, current_val_loss)
                
                if current_val_loss < bestValidationLoss
                    bestValidationLoss = current_val_loss
                    epochs_no_improve = 0
                    best_ann = deepcopy(ann)
                else
                    epochs_no_improve += 1
                end
            end

            if !isempty(testInputs)
                push!(testLosses, loss(ann, testInputs', testTargets'))
            end

            if epochs_no_improve >= maxEpochsVal || current_train_loss <= minLoss
                break
            end
        end

        # Load best model parameters if validation was used
        if !isempty(validationInputs)
            ann = best_ann
        end
        
        return (ann, trainingLosses, validationLosses, testLosses)
    end


    # ---------------------------------------------------------------------------
    # --- From: Unit 4.1 & 4.2 - Metrics ---
    # ---------------------------------------------------------------------------
    """
        confusionMatrix(outputs, targets; weighted=true)

    Calculates multiclass confusion matrix and associated metrics.
    `weighted=true` computes weighted averages for metrics, `false` uses macro average.
    Overloaded for different input types.
    """
    function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
        @assert size(outputs) == size(targets)

        L = size(targets, 2)
        if L == 1
            return confusionMatrix(vec(outputs), vec(targets))
        elseif L == 2
            error("Two-column matrices are invalid for multiclass confusion matrix.")
        end

        sens = fill(NaN, L)
        spec = fill(NaN, L)
        ppv = fill(NaN, L)
        npv = fill(NaN, L)
        f1 = fill(NaN, L)
        
        for i in 1:L
            t = targets[:, i]
            o = outputs[:, i]
            
            TP = sum(t .& o)
            TN = sum((.!t) .& (.!o))
            FP = sum((.!t) .& o)
            FN = sum(t .& (.!o))
            
            sens[i] = (TP + FN) == 0 ? 0.0 : TP / (TP + FN)
            spec[i] = (TN + FP) == 0 ? 0.0 : TN / (TN + FP)
            ppv[i]  = (TP + FP) == 0 ? 0.0 : TP / (TP + FP)
            npv[i]  = (TN + FN) == 0 ? 0.0 : TN / (TN + FN)
            f1[i]   = (ppv[i] + sens[i] == 0) ? 0.0 : 2 * ppv[i] * sens[i] / (ppv[i] + sens[i])
        end

        confMatrix = [sum(targets[:, i] .& outputs[:, j]) for i in 1:L, j in 1:L]
        
        acc = accuracy(outputs, targets)
        error_rate = 1 - acc

        counts = vec(sum(targets, dims=1))
        
        if weighted
            sens_agg = sum(sens .* counts) / sum(counts)
            spec_agg = sum(spec .* counts) / sum(counts)
            ppv_agg  = sum(ppv .* counts) / sum(counts)
            npv_agg  = sum(npv .* counts) / sum(counts)
            f1_agg   = sum(f1 .* counts) / sum(counts)
        else # macro
            sens_agg = mean(sens)
            spec_agg = mean(spec)
            ppv_agg  = mean(ppv)
            npv_agg  = mean(npv)
            f1_agg   = mean(f1)
        end
        
        return (acc, error_rate, sens_agg, spec_agg, ppv_agg, npv_agg, f1_agg, confMatrix)
    end

    function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
        bool_outputs = classifyOutputs(outputs, threshold=threshold)
        return confusionMatrix(bool_outputs, targets, weighted=weighted)
    end

    function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
        o_bool = oneHotEncoding(outputs, classes)
        t_bool = oneHotEncoding(targets, classes)
        return confusionMatrix(o_bool, t_bool, weighted=weighted)
    end

    function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
        classes = unique(vcat(targets, outputs))
        return confusionMatrix(outputs, targets, classes, weighted=weighted)
    end

    """
        printConfusionMatrix(outputs, targets; ...)

    Prints the results from the confusionMatrix function in a formatted way.
    """
    function _print_metrics(acc, err, sens, spec, ppv, npv, f1, cm; weighted::Bool)
        agg_type = weighted ? "weighted" : "macro"
        println("=== Multiclass Confusion Matrix ($(agg_type)) ===")
        @printf "Accuracy:     %.4f\n" acc
        @printf "Error rate:   %.4f\n" err
        @printf "Sensitivity:  %.4f\n" sens
        @printf "Specificity:  %.4f\n" spec
        @printf "PPV:          %.4f\n" ppv
        @printf "NPV:          %.4f\n" npv
        @printf "F1-score:     %.4f\n" f1
        println("Confusion Matrix (rows=true, cols=predicted):")
        display(cm)
        println()
    end

    function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
        res = confusionMatrix(outputs, targets; weighted=weighted)
        _print_metrics(res...; weighted=weighted)
    end

    function printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
        res = confusionMatrix(outputs, targets; threshold=threshold, weighted=weighted)
        _print_metrics(res...; weighted=weighted)
    end

    function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
        res = confusionMatrix(outputs, targets, classes; weighted=weighted)
        _print_metrics(res...; weighted=weighted)
    end

    function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
        res = confusionMatrix(outputs, targets; weighted=weighted)
        _print_metrics(res...; weighted=weighted)
    end


    # ---------------------------------------------------------------------------
    # --- From: Unit 5 - Unit 5 - Cross-validation
    # ---------------------------------------------------------------------------


    # =========================
    # Helper: Stratified K-Fold generator
    # =========================
    # Ensures that each fold has approximately the same class distribution.
    function stratified_kfold_indices(labels::AbstractVector, k::Int; rng=Random.default_rng())
        @assert k ≥ 2
        N = length(labels)
        folds = similar(collect(1:N), Int)  # vector of fold indices (1..k)

        # Group sample indices by class
        groups = Dict{eltype(labels), Vector{Int}}()
        for (i, y) in enumerate(labels)
            push!(get!(groups, y, Int[]), i)
        end

        # Shuffle indices within each class and assign folds in round-robin fashion
        for idxs in values(groups)
            Random.shuffle!(rng, idxs)
            for (pos, idx) in enumerate(idxs)
                folds[idx] = 1 + mod(pos - 1, k)
            end
        end
        return folds
    end

end  # module Utils