"""
network.jl
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries

using NumericFuns



type Network
    num_layers
    sizes
    biases
    weights
    
    function Network(sizes)
	"""The list ``sizes`` contains the number of neurons in the
	respective layers of the network.  For example, if the list
	was [2, 3, 1] then it would be a three-layer network, with the
	first layer containing 2 neurons, the second layer 3 neurons,
	and the third layer 1 neuron.  The biases and weights for the
	network are initialized randomly, using a Gaussian
	distribution with mean 0, and variance 1.  Note that the first
	layer is assumed to be an input layer, and by convention we
	won't set any biases for those neurons, since biases are only
	ever used in computing the outputs from later layers."""
        biases = [randn(n, 1) for n in sizes[2:end]]
        weights = [randn(n, m) for (m, n) in
                    zip(sizes[1:end-1], sizes[2:end])]
        new(length(sizes), sizes, biases, weights)
    end
end

function feedforward(net, a)
    for (b, w) in zip(net.biases, net.weights)
        a = sigmoid(w * a + b)
    end
    return a
end

function SGD(net, training_data, epochs, mini_batch_size, eta,
        test_data=None)
    """Train the neural network using mini-batch stochastic
    gradient descent.  The "training_data" is a list of tuples
    "(x, y)" representing the training inputs and the desired
    outputs.  The other non-optional parameters are
    self-explanatory.  If "test_data" is provided then the
    network will be evaluated against the test data after each
    epoch, and partial progress printed out.  This is useful for
    tracking progress, but slows things down substantially."""
    
    if test_data 
        n_test = lenght(test_data)
    end
    n = lenght(training_data)
    for j in 1:epochs
        shuffle(training_data)
        mini_batches = [
            training_data[k:k+mini_batch_size]
            for k in 1:mini_batch_size:n]
        for mini_batch in mini_batches
            update_mini_batch(net, mini_batch, eta)
        end
        if test_data
            println("Epoch $j: $(evaluate(net, test_data)) / $n_test")
        else
            println("Epoch $j complete")
        end
    end
end


function update_mini_batch(net, mini_batch, eta)
    """Update the network's weights and biases by applying
    gradient descent using backpropagation to a single mini batch.
    The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
    is the learning rate."""
    nabla_b = [zeros(b.shape) for b in net.biases]
    nabla_w = [zeros(w.shape) for w in net.weights]
    for (x, y) in mini_batch
        delta_nabla_b, delta_nabla_w = backprop(net, x, y)
        nabla_b = [nb+dnb for (nb, dnb) in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for (nw, dnw) in zip(nabla_w, delta_nabla_w)]
    end
    net.weights = [w-(eta/lenght(mini_batch))*nw
                    for (w, nw) in zip(net.weights, nabla_w)]
    net.biases = [b-(eta/lenght(mini_batch))*nb
                   for (b, nb) in zip(net.biases, nabla_b)]
end

function backprop(net, x, y)
    """Return a tuple ``(nabla_b, nabla_w)`` representing the
    gradient for the cost function C_x.  ``nabla_b`` and
    ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
    to ``net.biases`` and ``net.weights``."""
    nabla_b = [zeros(b.shape) for b in net.biases]
    nabla_w = [zeros(w.shape) for w in net.weights]
    # feedforward
    activation = x
    activations = [x] # list to store all the activations, layer by layer
    zs = [] # list to store all the z vectors, layer by layer
    for (b, w) in zip(net.biases, net.weights)
        z = w*activation+b
        zs.append!(z)
        activation = sigmoid(z)
        activations.append!(activation)
    end
    # backward pass
    delta = cost_derivative(activations[end], y) * sigmoid_prime(zs[end])
    nabla_b[end] = delta
    nabla_w[end] = delta*transpose(activations[end-1])
    # Note that the variable l in the loop below is used a little
    # differently to the notation in Chapter 2 of the book.  Here,
    # l = 1 means the last layer of neurons, l = 2 is the
    # second-last layer, and so on.  It's a renumbering of the
    # scheme in the book, used here to take advantage of the fact
    # that Python can use negative indices in lists.
    for l in range(1, net.num_layers-1)
        z = zs[end-l]
        sp = sigmoid_prime(z)
        delta = transpose(net.weights[end-l])*delta*sp
        nabla_b[-l] = delta
        nabla_w[-l] = dot(delta, activations[end-l-1])
    end
    return (nabla_b, nabla_w)
end

function evaluate(net, test_data)
    """Return the number of test inputs for which the neural
    network outputs the correct result. Note that the neural
    network's output is assumed to be the index of whichever
    neuron in the final layer has the highest activation."""
    test_results = [(maximum(feedforward(net, x)), y)
                    for (x, y) in test_data]
    return sum([int(x == y) for (x, y) in test_results])
end

function cost_derivative(output_activations, y)
    """Return the vector of partial derivatives \partial C_x /
    \partial a for the output activations."""
    return (output_activations-y)
end
    
function sigmoid_prime(z)
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
end
