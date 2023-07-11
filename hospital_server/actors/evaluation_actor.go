package actors

import (
	utils "agentske/hospital_server/proto_conversion"
	messages "agentske/proto"
	nn "agentske/training"
	"github.com/asynkron/protoactor-go/actor"
	"log"
)

type EvaluationActor struct {
	coordinationActor *actor.PID
}

func newEvaluationActor() actor.Actor {
	return &EvaluationActor{}
}

func (state *EvaluationActor) Receive(context actor.Context) {
	switch msg := context.Message().(type) {
	case *messages.EvaluationDataSets:
		log.Println("Evaluation Actor started:", context.Self().String())
		state.coordinationActor = context.Parent()
		Xv, Yv, _ := utils.GetDataSetsFromProto(msg.Validation)
		nn.StartEvaluation(Xv, Yv, context)
		context.Send(context.Parent(), &messages.EvaluationFinished{})

	case *actor.Stopped:
		log.Println("Evaluation Actor stopped:", context.Self().String())
	}
}
