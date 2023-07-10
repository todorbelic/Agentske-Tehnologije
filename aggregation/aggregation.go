package main

import (
	messages "agentske/proto"
	"agentske/training"
	"fmt"

	console "github.com/asynkron/goconsole"
	"github.com/asynkron/protoactor-go/actor"
	"github.com/asynkron/protoactor-go/remote"
	"gonum.org/v1/gonum/mat"
)

type AggregationActor struct{}

var n *training.MLP

func (*AggregationActor) Receive(context actor.Context) {
	switch msg := context.Message().(type) {
	case *messages.GetGlobalWeights:
		globalWeights := ConvertToGlobalWeights()
		context.Respond(globalWeights)
	case *messages.GradientUpdate:
		fmt.Println("update")
		for i, weightLayer := range msg.Weights {
			nw := mat.NewDense(len(weightLayer.Weights), 1, weightLayer.Weights)
			nb := mat.NewDense(len(weightLayer.Biases), 1, weightLayer.Biases)

			fmt.Println(nw.Dims())
			fmt.Println(nb.Dims())
			fmt.Println(i)
			// alpha := n.config.Eta / float64(N)

			// scalednw := new(mat.Dense)
			// scalednw.Scale(alpha, nw)

			// scalednb := new(mat.Dense)
			// scalednb.Scale(alpha, nb)

			// wprime := new(mat.Dense)
			// wprime.Sub(n.weights[i], scalednw)

			// bprime := new(mat.Dense)
			// bprime.Sub(n.biases[i], scalednb)

			// n.weights[i] = wprime
			// n.biases[i] = bprime
		}
	}
}

func ConvertToGlobalWeights() *messages.GlobalWeights {
	globalWeights := &messages.GlobalWeights{}

	// Convert biases
	for _, bias := range n.GetBiases() {
		biasData := &messages.Biases{
			Data: bias.RawMatrix().Data,
		}
		globalWeights.Biases = append(globalWeights.Biases, biasData)
	}

	// Convert weights
	for _, weight := range n.GetWeights() {
		weightData := &messages.Weights{
			Data: weight.RawMatrix().Data,
		}
		globalWeights.Weights = append(globalWeights.Weights, weightData)
	}

	return globalWeights
}

func main() {
	con := training.Config{
		Epochs:    25,
		Eta:       0.3,
		BatchSize: 32,
	}
	arch := []int{1600, 15, 8, 1}
	n = training.New(con, arch...)
	n.ReadWeightsFromFile("weights.json")

	fmt.Println("Read")

	system := actor.NewActorSystem()
	remoteConfig := remote.Configure("127.0.0.1", 8091)
	remoting := remote.NewRemote(system, remoteConfig)
	remoting.Start()

	// register a name for our local actor so that it can be spawned remotely
	remoting.Register("AggregationActor", actor.PropsFromProducer(func() actor.Actor { return &AggregationActor{} }))
	console.ReadLine()
}
