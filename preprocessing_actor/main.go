package main

import (
	"agentske/preprocessing"
	messages "agentske/proto"
	nn "agentske/training"
	"fmt"
	"log"
	"time"

	console "github.com/asynkron/goconsole"
	"github.com/asynkron/protoactor-go/actor"
	"github.com/asynkron/protoactor-go/remote"
	"gonum.org/v1/gonum/mat"
)

type PreprocessingActor struct {
	coordinationActor *actor.PID
}

type TrainingActor struct {
	coordinationActor *actor.PID
}

type CoordinationActor struct {
	trainingActor      *actor.PID
	preprocessingActor *actor.PID
	validationActor    *actor.PID
	aggregationActor   *actor.PID
}

func newCoordinationActor() actor.Actor {
	return &CoordinationActor{}
}

func newPreprocessingActor() actor.Actor {
	return &PreprocessingActor{}
}

func newTrainingActor() actor.Actor {
	return &TrainingActor{}
}

func (state *CoordinationActor) Receive(context actor.Context) {
	switch msg := context.Message().(type) {
	case *messages.ActivateLocalTraining:
		log.Println("Coordination Actor started:", context.Self().String())
		propsPreprocessing := actor.PropsFromProducer(newPreprocessingActor)
		preprocessingActor := context.Spawn(propsPreprocessing)
		state.preprocessingActor = preprocessingActor
		propsTraining := actor.PropsFromProducer(newTrainingActor)
		trainingActor := context.Spawn(propsTraining)
		state.trainingActor = trainingActor
		state.aggregationActor = msg.AggregationActor
		context.Send(preprocessingActor, &messages.ActivatePreproc{Sender: msg.Sender})
	case *messages.GetTrainingActor:
		context.Respond(state.trainingActor)

	case *messages.GetAggregationActor:
		context.Respond(state.aggregationActor)
	}

}

func getTrainingDataFromProto(data *messages.Data) (*mat.Dense, *mat.Dense, error) {
	rows, cols := len(data.Histograms), len(data.Histograms[0].Values)
	trainingData := make([]float64, rows*cols)
	for i, hist := range data.Histograms {
		copy(trainingData[i*cols:(i+1)*cols], hist.Values)
	}
	X := mat.NewDense(rows, cols, trainingData)
	Y := mat.NewDense(rows, 1, data.Labels)
	return X, Y, nil
}

func (state *TrainingActor) Receive(context actor.Context) {
	switch msg := context.Message().(type) {
	case *messages.DataSets:
		log.Println("Training Actor started:", context.Self().String())
		state.coordinationActor = context.Parent()
		X, Y, _ := getTrainingDataFromProto(msg.Training)
		Xv, Yv, _ := getTrainingDataFromProto(msg.Validation)
		fmt.Println(X.Dims())
		nn.StartTraining(X, Y, Xv, Yv, context, state.coordinationActor)
	}
}

func ConvertToProtoData(data preprocessing.Data) (*messages.Data, error) {
	protoData := &messages.Data{}

	// Assign Labels
	protoData.Labels = ConvertToInt32Slice(data.Labels)

	// Convert Histograms
	for _, hist := range data.Histograms {
		protoHist := &messages.Histogram{}
		protoHist.Values = hist

		protoData.Histograms = append(protoData.Histograms, protoHist)
	}

	return protoData, nil
}

func ConvertToInt32Slice(labels []float64) []float64 {
	result := make([]float64, len(labels))
	for i, val := range labels {
		result[i] = float64(val)
	}
	return result
}

func (state *PreprocessingActor) Receive(context actor.Context) {
	switch msg := context.Message().(type) {
	case *messages.ActivatePreproc:
		log.Println("Preprocessing Actor started:", context.Self().String())
		state.coordinationActor = msg.Sender
		train, val := preprocessing.PreprocessImages()
		trainProto, _ := ConvertToProtoData(train)
		valProto, _ := ConvertToProtoData(val)
		message := &messages.DataSets{Training: trainProto, Validation: valProto}
		future, _ := context.RequestFuture(state.coordinationActor, &messages.GetTrainingActor{}, 1*time.Second).Result()
		pid, _ := future.(*actor.PID)
		context.Send(pid, message)
	}
}

func main() {
	actorSystem := actor.NewActorSystem()
	decider := func(reason interface{}) actor.Directive {
		fmt.Println("handling failure for child")
		return actor.StopDirective
	}
	cfg := remote.Configure("127.0.0.1", 8100)
	r := remote.NewRemote(actorSystem, cfg)
	r.Start()

	supervisor := actor.NewOneForOneStrategy(10, 1000, decider)
	rootContext := actorSystem.Root

	props := actor.PropsFromProducer(newCoordinationActor, actor.WithSupervisor(supervisor))

	pid := rootContext.Spawn(props)
	spawnResponse, err := r.SpawnNamed("127.0.0.1:8091", "AggregationActor", "AggregationActor", time.Second)

	if err != nil {
		panic(err)
		return
	}

	// get spawned PID
	rootContext.Send(pid, &messages.ActivateLocalTraining{Sender: pid, AggregationActor: spawnResponse.Pid})
	console.ReadLine()
}
