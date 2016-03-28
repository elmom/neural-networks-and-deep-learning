@doc "
network.jl
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"

module NN

#### Libraries
using NumericFuns
using Iterators


type Network
    num_layers
    sizes
    biases
    weights

    function Network(sizes)
        "The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."
        biases = [randn(n, 1) for n in sizes[2:end]]
        weights = [randn(n, m) for (m, n) in
                    zip(sizes[1:end-1], sizes[2:end])]
        new(length(sizes), sizes, biases, weights)
    end
end

function feedforward(net, a)
    for (b, w) in zip(net.biases, net.weights)
        a = σ(w * a + b)
    end
    return a
end

function SGD(net, training_data, epochs, mini_batch_size, η;
        test_data=nothing)
    """Train the neural network using mini-batch stochastic
    gradient descent.  The "training_data" is a list of tuples
    "(x, y)" representing the training inputs and the desired
    outputs.  The other non-optional parameters are
    self-explanatory.  If "test_data" is provided then the
    network will be evaluated against the test data after each
    epoch, and partial progress printed out.  This is useful for
    tracking progress, but slows things down substantially."""

    report_progress(net, 0, test_data)
    n = length(training_data)
    for j in 1:epochs
        @time begin
            shuffle(training_data)
            #mini_batches = [
            #    training_data[k:k+mini_batch_size-1]
            #    for k in 1:mini_batch_size:n-1]
            mini_batches = partition(training_data, mini_batch_size)
            for mini_batch in mini_batches
                update_mini_batch(net, mini_batch, η)
            end
        end
        report_progress(net, j, test_data)
    end
end

function report_progress(net, j, test_data=nothing)
    if test_data != nothing
        n_test = length(test_data)
    end
    if test_data != nothing
        @time println("Epoch $j: $(evaluate(net, test_data)) / $n_test")
    else
        println("Epoch $j complete")
    end
end

init_∇(c) = [zeros(size(b)) for b in c]

function update_mini_batch(net, mini_batch, η)
    "Update the network's weights and biases by applying
    gradient descent using backpropagation to a single mini batch.
    The ``mini_batch`` is a list of tuples ``(x, y)``, and ``η``
    is the learning rate."
    ∇b = init_∇(net.biases)
    ∇w = init_∇(net.weights)
    batch_size = length(mini_batch)
    for (x, y) in mini_batch
        δ∇b, δ∇w = backprop(net, x, y)
        #@code_warntype backprop(net, x, y)
        ∇b = [nb+dnb for (nb, dnb) in zip(∇b, δ∇b)]
        ∇w = [nw+dnw for (nw, dnw) in zip(∇w, δ∇w)]
    end
    net.weights = [w-(η/batch_size)*nw
                    for (w, nw) in zip(net.weights, ∇w)]
    net.biases = [b-(η/batch_size)*nb
                   for (b, nb) in zip(net.biases, ∇b)]
end

function backprop(net, x, y)
    "Return a tuple ``(∇b, ∇w)`` representing the
    gradient for the cost function C_x.  ``∇b`` and
    ``∇w`` are layer-by-layer lists of numpy arrays, similar
    to ``net.biases`` and ``net.weights``."
    ∇b = init_∇(net.biases)
    ∇w = init_∇(net.weights)
    # feedforward
    activation = x
    activations = [] # list to store all the activations, layer by layer
    push!(activations, activation)
    zs = [] # list to store all the z vectors, layer by layer
    for (b, w) in zip(net.biases, net.weights)
        z = w*activation+b
        push!(zs, z)
        activation = σ(z)
        push!(activations, activation)
    end
    # backward pass
    δ = cost_derivative(activations[end], y) .* σ_prime(zs[end])
    ∇b[end] = δ
    ∇w[end] = δ * transpose(activations[end-1])
    # Note that the variable l in the loop below is used a little
    # differently to the notation in Chapter 2 of the book.  Here,
    # l = 1 means the last layer of neurons, l = 2 is the
    # second-last layer, and so on.  It's a renumbering of the
    # scheme in the book, used here to take advantage of the fact
    # that Python can use negative indices in lists.
    for l in 1:net.num_layers-2
        z = zs[end-l]
        sp = σ_prime(z)
        δ = transpose(net.weights[end-l+1])*δ.*sp
        ∇b[end-l] = δ
        ∇w[end-l] = δ * transpose(activations[end-l-1])
    end
    return (∇b, ∇w)
end

function evaluate(net, test_data)
    "Return the number of test inputs for which the neural
    network outputs the correct result. Note that the neural
    network's output is assumed to be the index of whichever
    neuron in the final layer has the highest activation."
    test_results = [(indmax(feedforward(net, x))-1, y)
                    for (x, y) in test_data]
    return sum([Int(x == y) for (x, y) in test_results])
end

function cost_derivative(output_activations, y)
    "Return the vector of partial derivatives \partial C_x /
    \partial a for the output activations."
    return (output_activations-y)
end

function sigmoid2(z)
    """The sigmoid function."""
    return 1.0./(1.0+exp(-z))
end

σ = sigmoid

function σ_prime(z)
    "Derivative of the sigmoid function."
    return σ(z) .* (1-σ(z))
end

end
