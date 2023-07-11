package main

import (
	actors "agentske/hospital_server/actors"
	messages "agentske/proto"
	"fmt"
	"github.com/asynkron/protoactor-go/actor"
	"github.com/asynkron/protoactor-go/remote"
	"net/http"
	"time"
)

func main() {
	http.HandleFunc("/training", handleTraining)
	http.HandleFunc("/evaluation", handleEvaluation)
	http.ListenAndServe(":8080", nil)
}

func handleTraining(w http.ResponseWriter, r *http.Request) {
	// Create a new actor system for each request
	actorSystem := actor.NewActorSystem()
	decider := func(reason interface{}) actor.Directive {
		fmt.Println("handling failure for child")
		return actor.StopDirective
	}
	cfg := remote.Configure("127.0.0.1", 8100)
	actors.Remote = remote.NewRemote(actorSystem, cfg)
	actors.Remote.Start()

	supervisor := actor.NewOneForOneStrategy(20, 1000, decider)
	rootContext := actorSystem.Root

	props := actor.PropsFromProducer(actors.NewCoordinationActor, actor.WithSupervisor(supervisor))

	pid := rootContext.Spawn(props)
	spawnResponse, err := actors.Remote.SpawnNamed("127.0.0.1:8091", "AggregationActor", "AggregationActor", time.Second)

	if err != nil {
		panic(err)
		return
	}
	// Send a message to the actor
	rootContext.Send(pid, &messages.ActivateLocalTraining{AggregationActor: spawnResponse.Pid})

	fmt.Fprintln(w, "Request processed")
}

func handleEvaluation(w http.ResponseWriter, r *http.Request) {
	// Create a new actor system for each request
	actorSystem := actor.NewActorSystem()
	decider := func(reason interface{}) actor.Directive {
		fmt.Println("handling failure for child")
		return actor.StopDirective
	}
	cfg := remote.Configure("127.0.0.1", 8100)
	actors.Remote = remote.NewRemote(actorSystem, cfg)
	actors.Remote.Start()

	supervisor := actor.NewOneForOneStrategy(20, 1000, decider)
	rootContext := actorSystem.Root

	props := actor.PropsFromProducer(actors.NewCoordinationActor, actor.WithSupervisor(supervisor))

	pid := rootContext.Spawn(props)
	spawnResponse, err := actors.Remote.SpawnNamed("127.0.0.1:8091", "AggregationActor", "AggregationActor", time.Second)

	if err != nil {
		panic(err)
		return
	}
	// Send a message to the actor
	rootContext.Send(pid, &messages.ActivateEvaluation{AggregationActor: spawnResponse.Pid})

	fmt.Fprintln(w, "Request processed")
}
