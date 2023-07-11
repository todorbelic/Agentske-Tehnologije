package actors

import (
	utils "agentske/hospital_server/proto_conversion"
	messages "agentske/proto"
	nn "agentske/training"
	"github.com/asynkron/protoactor-go/actor"
	"log"
)

type TrainingActor struct {
	coordinationActor *actor.PID
}

func newTrainingActor() actor.Actor {
	return &TrainingActor{}
}

func (state *TrainingActor) Receive(context actor.Context) {
	switch msg := context.Message().(type) {
	case *messages.TrainingDataSets:
		log.Println("Training Actor started:", context.Self().String())
		state.coordinationActor = context.Parent()
		X, Y, _ := utils.GetDataSetsFromProto(msg.Training)
		Xv, Yv, _ := utils.GetDataSetsFromProto(msg.Validation)
		nn.StartTraining(X, Y, Xv, Yv, context)
		context.Send(context.Parent(), &messages.TrainingFinished{})

	case *actor.Stopped:
		log.Println("Training Actor stopped:", context.Self().String())
	}
}
