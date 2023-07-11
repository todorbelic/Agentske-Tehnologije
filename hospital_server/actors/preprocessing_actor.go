package actors

import (
	utils "agentske/hospital_server/proto_conversion"
	"agentske/preprocessing"
	messages "agentske/proto"
	"github.com/asynkron/protoactor-go/actor"
	"log"
	"time"
)

type PreprocessingActor struct {
	coordinationActor *actor.PID
}

func newPreprocessingActor() actor.Actor {
	return &PreprocessingActor{}
}

func (state *PreprocessingActor) Receive(context actor.Context) {
	switch context.Message().(type) {
	case *messages.ActivatePreprocTraining:
		log.Println("Preprocessing Actor started:", context.Self().String())
		state.coordinationActor = context.Parent()
		train, val := preprocessing.PreprocessImagesForTraining()
		trainProto, _ := utils.ConvertToProtoData(train)
		valProto, _ := utils.ConvertToProtoData(val)
		message := &messages.TrainingDataSets{Training: trainProto, Validation: valProto}
		future, _ := context.RequestFuture(state.coordinationActor, &messages.GetTrainingActor{}, 1*time.Second).Result()
		pid, _ := future.(*actor.PID)
		context.Send(pid, message)
		context.Send(context.Parent(), &messages.PreprocessingFinished{})

	case *messages.ActivatePreprocEvaluation:
		log.Println("Preprocessing Actor started:", context.Self().String())
		state.coordinationActor = context.Parent()
		val := preprocessing.PreprocessImagesForEvaluation()
		valProto, _ := utils.ConvertToProtoData(val)
		message := &messages.EvaluationDataSets{Validation: valProto}
		future, _ := context.RequestFuture(state.coordinationActor, &messages.GetEvaluationActor{}, 1*time.Second).Result()
		pid, _ := future.(*actor.PID)
		context.Send(pid, message)
		context.Send(context.Parent(), &messages.PreprocessingFinished{})

	case *actor.Stopped:
		log.Println("Preprocessing Actor stopped:", context.Self().String())
	}
}
