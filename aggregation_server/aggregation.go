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
	switch msg := context.Message().(type) {
	case *messages.GetGlobalWeights:
		globalWeights := training.ConvertToGlobalWeights(n)
		context.Respond(globalWeights)
	case *messages.GradientUpdate:
		training.UpdateGlobalWeights(n, msg)
	}
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
