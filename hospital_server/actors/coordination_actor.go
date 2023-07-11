package actors

import (
	messages "agentske/proto"
	"github.com/asynkron/protoactor-go/actor"
	"github.com/asynkron/protoactor-go/remote"
	"log"
)

var Remote *remote.Remote

type CoordinationActor struct {
	trainingActor      *actor.PID
	preprocessingActor *actor.PID
	evaluationActor    *actor.PID
	aggregationActor   *actor.PID
}

func NewCoordinationActor() actor.Actor {
	return &CoordinationActor{}
}

func (state *CoordinationActor) Receive(context actor.Context) {
	switch msg := context.Message().(type) {
	case *messages.ActivateLocalTraining:
		log.Println("Coordination Actor started:", context.Self().String())
		//spawn preprocessing actor
		propsPreprocessing := actor.PropsFromProducer(newPreprocessingActor)
		preprocessingActor := context.Spawn(propsPreprocessing)
		state.preprocessingActor = preprocessingActor
		//spawn training actor
		propsTraining := actor.PropsFromProducer(newTrainingActor)
		trainingActor := context.Spawn(propsTraining)
		state.trainingActor = trainingActor
		state.aggregationActor = msg.AggregationActor
		//start preprocessing
		context.Send(preprocessingActor, &messages.ActivatePreprocTraining{})

	case *messages.ActivateEvaluation:
		log.Println("Coordination Actor started:", context.Self().String())
		//spawn preprocessing actor
		propsPreprocessing := actor.PropsFromProducer(newPreprocessingActor)
		preprocessingActor := context.Spawn(propsPreprocessing)
		state.preprocessingActor = preprocessingActor
		//spawn evaluation actor
		propsEvaluation := actor.PropsFromProducer(newEvaluationActor)
		evaluationActor := context.Spawn(propsEvaluation)
		state.evaluationActor = evaluationActor
		state.aggregationActor = msg.AggregationActor
		//start preprocessing
		context.Send(preprocessingActor, &messages.ActivatePreprocEvaluation{})

	case *messages.GetTrainingActor:
		context.Respond(state.trainingActor)

	case *messages.GetAggregationActor:
		context.Respond(state.aggregationActor)

	case *messages.GetEvaluationActor:
		context.Respond(state.evaluationActor)

	case *messages.PreprocessingFinished:
		context.Stop(state.preprocessingActor)

	case *messages.TrainingFinished:
		context.Stop(state.trainingActor)
		Remote.Shutdown(true)

	case *messages.EvaluationFinished:
		context.Stop(state.evaluationActor)
		Remote.Shutdown(true)
	}
}
