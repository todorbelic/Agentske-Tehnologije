package main

import (
	messages "agentske/proto"
	"agentske/training"
	"fmt"

	console "github.com/asynkron/goconsole"
	"github.com/asynkron/protoactor-go/actor"
	"github.com/asynkron/protoactor-go/remote"
)

type AggregationActor struct{}

var n *training.MLP

func (*AggregationActor) Receive(context actor.Context) {
	switch context.Message().(type) {
	case *messages.GetGlobalWeights:
		fmt.Println("stiglo")
		globalWeights := ConvertToGlobalWeights()
		fmt.Println(globalWeights)
		context.Respond(&globalWeights)
		context.Respond(&messages.GlobalWeightsTest{String_: "weights"})
	case *messages.GradientUpdate:
		fmt.Println("update")

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
