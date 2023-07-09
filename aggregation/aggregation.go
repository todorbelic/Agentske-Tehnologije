package main

import (
	messages "agentske/proto"
	console "github.com/asynkron/goconsole"
	"github.com/asynkron/protoactor-go/actor"
	"github.com/asynkron/protoactor-go/remote"
)

type AggregationActor struct{}

func (*AggregationActor) Receive(context actor.Context) {
	switch context.Message().(type) {
	case *messages.GetGlobalWeights:
		//fmt.Println("stiglo")
		context.Respond(&messages.GlobalWeights{String_: "weights"})
	}
}

func main() {

	system := actor.NewActorSystem()
	remoteConfig := remote.Configure("127.0.0.1", 8091)
	remoting := remote.NewRemote(system, remoteConfig)
	remoting.Start()

	// register a name for our local actor so that it can be spawned remotely
	remoting.Register("AggregationActor", actor.PropsFromProducer(func() actor.Actor { return &AggregationActor{} }))
	console.ReadLine()
}
